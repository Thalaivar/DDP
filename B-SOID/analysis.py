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

FEATS_TYPE = 'dis'
STRAINS = ["LL6-B2B", "LL5-B2B", "LL4-B2B", "LL3-B2B", "LL2-B2B", "LL1-B2B"]
DATASETS = ["strain-survey-batch-2019-05-29-e/", "strain-survey-batch-2019-05-29-d/", 
            "strain-survey-batch-2019-05-29-c/", "strain-survey-batch-2019-05-29-b/",
            "strain-survey-batch-2019-05-29-a/"]

BASE_DIR = 'D:/IIT/DDP/data'
MICE_DIR = BASE_DIR + '/analysis/mice'
RAW_DIR = BASE_DIR + '/raw'

BEHAVIOUR_LABELS = {
    'CCW-Turn': 0,
    'Groom-1': 1,
    'N/A': [2, 4, 10, 11, 13],
    'Rear-1': 3,
    'Groom-2': 5,
    'CW-Turn': 6,
    'Rear-CW-Turn': 7,
    'Locomote-1': 8,
    'Groom-3': 9,
    'Groom-4': 12,
}

try:
    os.makedirs(RAW_DIR)
except FileExistsError:
    pass

FPS = 30
STRIDE_WINDOW = 3

class Mouse:
    def __init__(self, metadata: dict):
        self.strain = metadata['Strain']
        self.filename = metadata['NetworkFilename']
        self.sex = metadata['Sex']
        self.id = metadata['MouseID']

        self.save_dir = MICE_DIR + '/' + self.strain.replace('/', '-') + '/' + self.id
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
        feats = self.load_features()
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
        strain, mouse_id = metadata['Strain'].replace('/', '-'), metadata['MouseID']
        save_dir = f'{MICE_DIR}/{strain}/{mouse_id}'

        with open(f'{save_dir}/metadata.info', 'rb') as f:
            metadata = joblib.load(f)
        
        mouse = Mouse(metadata)
        return mouse

    def load_features(self):
        if not os.path.isfile(f'{self.save_dir}/feats.sav'):
            feats = self.extract_features(features_type=FEATS_TYPE)
        else:
            with open(f'{self.save_dir}/feats.sav', 'rb') as f:
                feats = joblib.load(f)
        return feats

"""
    Helper functions for carrying out analysis per mouse:
        - transition_matrix_from_assay : calculates the transition matrix for a given assay
        - get_behaviour_info_from_assay : extracts statistics of all behaviours from given assay
"""
def transition_matrix_from_assay(mouse: Mouse, labels):
    n_lab = labels.max() + 1
    tmat = np.zeros((n_lab, n_lab))
    curr_lab = labels[0]
    for i in range(1, labels.size):
        lab = labels[i]
        tmat[curr_lab, lab] += 1
        curr_lab = lab
    
    for i in range(n_lab):
        tmat[i] /= tmat[i].sum()    

    return tmat

def behaviour_proportion(labels):
    n_lab = labels.max() + 1
    
    prop = [0 for _ in range(n_lab)]
    for i in range(labels.size):
        prop[labels[i]] += 1
    
    prop = np.array(prop)
    prop = prop/prop.sum()

    return prop

def get_behaviour_info_from_assay(mouse: Mouse, labels, behaviour_idx, min_bout_len):
    if not isinstance(behaviour_idx, list):
        behaviour_idx = [behaviour_idx]

    for i in range(labels.size):
        if labels[i] in behaviour_idx:
            labels[i] = -1

    total_duration, average_bout_len, n_bouts = 0, [], 0 
    i = 0
    while i < len(labels) - 1:
        if labels[i] < 0:
            total_duration += 1
            curr_idx = labels[i]
            curr_bout_len, j = 0, i + 1
            while curr_idx == labels[j] and j < len(labels) - 1:
                total_duration += 1
                curr_idx = labels[j]
                curr_bout_len += 1
                j += 1
        
            if curr_bout_len > min_bout_len:
                n_bouts += 1
                average_bout_len.append(curr_bout_len)
        
            i = j
        
        else:
            i += 1

    average_bout_len = sum(average_bout_len)/len(average_bout_len)

    # get durations in seconds
    total_duration = total_duration / FPS
    average_bout_len = average_bout_len / FPS

    return total_duration, n_bouts, average_bout_len

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
        transition_matrix_from_assay(mouse, labels)

    if parallel:
        Parallel(n_jobs=-1)(delayed(calculate_tmat)(i, data, clf) for i in range(N))
    else:
        for i in tqdm(range(N)):
            calculate_tmat(i, data, clf)

def calculate_behaviour_usage(data_lookup_file, parallel=True):
    clf_file = f'{BASE_DIR}/output/dis_classifiers.sav'
    with open(clf_file, 'rb') as f:
        clf = joblib.load(f)
    
    data = pd.read_csv(clf_file)
    N = data.shape[0]

    def behaviour_usage(i, data, clf):
        metadata = dict(data.iloc[i])
        mouse = Mouse(metadata)

        labels = mouse.get_behaviour_labels(clf)
        return behaviour_proportion(labels)
    
    if parallel:
        prop = Parallel(n_jobs=-1)(delayed(behaviour_usage)(i, data, clf) for i in range(N))
    else:
        prop = [behaviour_usage(i, data, clf) for i in range(N)]
    
    prop = np.vstack(prop)
    return prop.sum(axis=0)/prop.shape[0]

def calculate_behaviour_info_for_all_strains(data_lookup_file, min_bout_len):
    clf_file = f'{BASE_DIR}/output/dis_classifiers.sav'
    with open(clf_file, 'rb') as f:
        clf = joblib.load(f)
    
    data = pd.read_csv(data_lookup_file)
    N = data.shape[0]

    info = {
        'Strain': [], 
        'Sex': [], 
        'Total Duration':  [], 
        'Average Bout Length': [], 
        'No. of Bouts': []
    }

    for i in tqdm(range(N)):
        metadata = dict(data.iloc[i])
        mouse = Mouse(metadata)

        idx = [BEHAVIOUR_LABELS[key] for key in BEHAVIOUR_LABELS.keys() if key[:5] == 'Groom']
        labels = mouse.get_behaviour_labels(clf)
        total_duration, n_bouts, avg_bout_len = get_behaviour_info_from_assay(mouse, labels, idx, min_bout_len)

        info['Strain'].append(mouse.strain)
        info['Sex'].append(mouse.sex)
        info['Total Duration'].append(total_duration)
        info['Average Bout Length'].append(avg_bout_len)
        info['No. of Bouts'].append(n_bouts)
    
    return pd.DataFrame.from_dict(info)

if __name__ == "__main__":
    extract_features_per_mouse('bsoid_strain_data.csv')