import os
import joblib
import ftplib
import h5py
import numpy as np
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
from joblib import Parallel, delayed
from BSOID.data import _bsoid_format
from BSOID.preprocessing import likelihood_filter
from BSOID.prediction import *

STRAINS = ["LL6-B2B", "LL5-B2B", "LL4-B2B", "LL3-B2B", "LL2-B2B", "LL1-B2B"]
DATASETS = ["strain-survey-batch-2019-05-29-e/", "strain-survey-batch-2019-05-29-d/", 
            "strain-survey-batch-2019-05-29-c/", "strain-survey-batch-2019-05-29-b/",
            "strain-survey-batch-2019-05-29-a/"]

BASE_DIR = 'D:/IIT/DDP/data'
MICE_DIR = BASE_DIR + '/analysis/mice'
RAW_DIR = BASE_DIR + '/raw'

try:
    os.makedirs(RAW_DIR)
except FileExistsError:
    pass

FPS = 30
STRIDE_WINDOW = 3

class Mouse:
    def __init__(self, metadata: dict):
        self.strain = metadata['Strain'].replace('/', '-')
        self.filename = metadata['NetworkFilename']
        self.sex = metadata['Sex']
        self.id = metadata['MouseID']

        self.save_dir = f'{MICE_DIR}/{self.strain}/{self.id}'
        try:
            os.makedirs(self.save_dir)
        except:
            pass
        
        # dump all metadata for future reference
        self.metadata = metadata
        self.save()

    def extract_features(self, features_type='dis'):
        data = self._get_raw_data()
        if features_type == 'dis':
            from BSOID.features.displacement_feats import extract_feats, window_extracted_feats
        
        feats = frameshift_features(data, STRIDE_WINDOW, FPS, extract_feats, window_extracted_feats)

        with open(f'{self.save_dir}/feats.sav', 'wb') as f:
            joblib.dump(feats, f)
        
        return feats

    def get_behaviour_labels(self, clf):
        if os.path.isfile(f'{self.save_dir}/feats.npy'):
            feats = self.load_features()
        else:
            feats = self.extract_features()
        labels = frameshift_predict(feats, clf, STRIDE_WINDOW)
        return labels

    def _get_raw_data(self):
        _, _, movie_name = self.filename.split('/')
        filename = f'{RAW_DIR}/{movie_name[0:-4]}_pose_est_v2.h5'
        f = h5py.File(filename, 'r')
        data = list(f.keys())[0]
        keys = list(f[data].keys())
        conf, pos = np.array(f[data][keys[0]]), np.array(f[data][keys[1]])
        f.close()

        # trim start and end
        end_trim = FPS*2*60
        conf, pos = conf[end_trim:-end_trim,:], pos[end_trim:-end_trim,:]
        data = _bsoid_format(conf, pos)
        fdata, perc_filt = likelihood_filter(data)

        if perc_filt > 10:
            logging.warn(f'mouse:{self.strain}/{self.id}: % data filtered from raw data is too high ({perc_filt} %)')
        
        data_shape = fdata['x'].shape
        logging.info(f'preprocessed raw data of shape: {data_shape}')

        return fdata
    
    def save(self):
        with open(f'{self.save_dir}/metadata.info', 'wb') as f:
            joblib.dump(self.metadata, f)
    
    @staticmethod
    def load(metadata: dict):
        strain, mouse_id = metadata['strain'], metadata['mouse_id']
        save_dir = f'{MICE_DIR}/{strain}/{mouse_id}'

        with open(f'{save_dir}/metadata.info', 'wb') as f:
            metadata = joblib.load(f)
        
        mouse = Mouse(metadata)
        return mouse

    def load_features(self):
        with open(f'{self.save_dir}/feats.sav', 'rb') as f:
            feats = joblib.load(f)
        return feats
        
"""
    Helper functions for carrying out analysis per mouse:
        - transition_matrix_from_assay : calculates the transition matrix for a given assay
        - get_behaviour_info_from_assay : extracts statistics of all behaviours from given assay
"""
def transition_matrix_from_assay(mouse: Mouse, labels, savefile=None):
    n_lab = labels.max() + 1
    tmat = np.zeros((n_lab, n_lab))
    curr_lab = labels[0]
    for i in range(1, labels.size):
        lab = labels[i]
        tmat[curr_lab, lab] += 1
        curr_lab = lab
    
    for i in range(n_lab):
        tmat[i] /= tmat[i].sum()    
    
    if savefile is not None:
        np.save(f'{mouse.save_dir}/{savefile}', tmat)

    return tmat

def get_behaviour_info_from_assay(mouse: Mouse, labels, min_bout_len):
    n_lab = labels.max() + 1

    # total behaviour duration
    total_duration = [0 for _ in range(n_lab)]
    for i in range(len(labels)):
        total_duration[labels[i]] += 1
    
    # no. of bouts and their average length
    n_bouts = [0 for _ in range(n_lab)]
    bout_lens = [[] for _ in range(n_lab)]
    i = 0
    while i < len(labels):
        curr_idx = labels[i]
        curr_bout_len, j = 0, i + 1
        while curr_idx == labels[j] and j < len(labels):
            curr_idx = labels[j]
            curr_bout_len += 1
            j += 1
        
        if curr_bout_len > min_bout_len:
            n_bouts[labels[i]] += 1
            bout_lens[labels[i]] = curr_bout_len
        
        i = j

    avg_bout_lens = [sum(x)/len(x) for x in bout_lens]

    # get durations in seconds
    total_duration = np.array(total_duration) / FPS
    avg_bout_lens = np.array(avg_bout_lens) / FPS

    with open(f'{mouse.save_dir}/behaviour_info.sav', 'wb') as f:
        joblib.dump([total_duration, np.array(n_bouts), avg_bout_lens], f)

    return total_duration, np.array(n_bouts), avg_bout_lens

"""
    modules to run different analyses for all mice
"""
def extract_features_per_mouse(data_lookup_file):
    data = pd.read_csv(data_lookup_file)
    N = data.shape[0]

    print(f'extracting raw data for {N} mice')

    def extract(i, data):
        mouse_data = dict(data.iloc[i])
        mouse = Mouse(mouse_data)
        if os.path.isfile(f'{mouse.save_dir}/feats.sav'):
            print(f'skipping {mouse.save_dir}')
            pass
        else:
            mouse.extract_features()

    Parallel(n_jobs=-1)(delayed(extract)(i, data) for i in range(N))

    # validate that all mice were included
    total_mice = 0
    strains = os.listdir(MICE_DIR)
    for strain in strains:
        ids = os.listdir(f'{MICE_DIR}/{strain}')
        total_mice += len(ids)
    
    assert total_mice == N, 'some mice were overwritten'

def calculate_transition_matrix_for_entire_assay(data_lookup_file, parallel=True):
    clf_file = f'{BASE_DIR}/output/dis_classifiers.sav'
    with open(clf_file, 'rb') as f:
        clf = joblib.load(f)
    
    data = pd.read_csv(data_lookup_file)
    N = data.shape[0]

    print(f'calculating transition matrix for full assay of {N} mice')
    def calculate_tmat(i, data, clf):
        # load mouse from metadata 
        metadata = dict(data.iloc[i])
        mouse = Mouse.load(metadata)

        # get behaviour labels for entire assay
        labels = mouse.get_behaviour_labels(clf)
        transition_matrix_from_assay(mouse, labels, savefile='full_assay_tmat')

    if parallel:
        Parallel(n_jobs=-1)(delayed(calculate_tmat)(i, data, clf) for i in range(N))
    else:
        for i in tqdm(range(N)):
            calculate_tmat(i, data, clf)

def calculate_behaviour_usage(data_lookup_file,  parallel=True):
    data = pd.read_csv(data_lookup_file)
    
    N = data.shape[0]

    clf 
    def get_behaviour_prop(i, data, clf):
