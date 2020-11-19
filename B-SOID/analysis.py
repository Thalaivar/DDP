import os
import joblib
import ftplib
import h5py
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from BSOID.data import _bsoid_format
from BSOID.preprocessing import likelihood_filter

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
TEMPORAL_WINDOW = None
TEMPORAL_DIMS = None

class Mouse:
    def __init__(self, metadata: dict):
        self.strain = metadata['strain']
        self.filename = metadata['network_filename']
        self.sex = metadata['sex']
        self.id = metadata['mouse_id']

        self.save_dir = f'{MICE_DIR}/{self.strain}/{self.id}'
        try:
            os.makedirs(self.save_dir)
        except:
            pass
        
        # dump all metadata for future reference
        self.metadata = metadata
        self.save()

    def extract_features(self, window_feats=True, features_type='dis'):
        data = self._get_raw_data()
        if features_type == 'dis':
            from BSOID.features.displacement_feats import extract_feats, window_extracted_feats
        
        feats = extract_feats(data, FPS)
        if window_feats:
            feats = window_extracted_feats([feats], STRIDE_WINDOW, TEMPORAL_WINDOW, TEMPORAL_DIMS)[0]
            
        logging.info(f'extracted features of shape {feats.shape}')

        np.save(f'{self.save_dir}/feats.npy', feats)
        return feats

    def get_behaviour_labels(self, clf):
        if os.path.isfile(f'{self.save_dir}/feats.npy'):
            feats = self.load_features()
        else:
            feats = self.extract_features()
        labels = clf.predict(feats)
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
        return np.load(f'{self.save_dir}/feats.npy')

def extract_features_per_mouse(data_lookup_file):
    import pandas as pd
    data = pd.read_csv(data_lookup_file)
    N = data.shape[0]

    logging.info(f'extracting raw data for {N} mice')

    def extract(i, data):
        mouse_data = dict(data.iloc[i])
        mouse = Mouse(**mouse_data)
        if os.path.exists(f'{mouse.save_dir}/feats.npy'):
            pass
        else:
            mouse.extract_features()
            mouse.save()

    from joblib import Parallel, delayed
    Parallel(n_jobs=-1)(delayed(extract)(i, data) for i in range(N))
        
def transition_matrix_full_assay(mouse: Mouse, clf):
    labels = mouse.get_behaviour_labels(clf)
    
    n_lab = labels.max() + 1
    t_mat_full = np.zeros((n_lab, n_lab))
    curr_lab = labels[0]
    for i in range(1, labels.size):
        lab = labels[i]
        t_mat_full[curr_lab, lab] += 1
        curr_lab = lab
    
    for i in range(n_lab):
        t_mat_full[i] /= t_mat_full[i].sum()    
    
    np.save(f'{mouse.save_dir}/transition_matrix_full.npy', t_mat_full)

    return t_mat_full

def 
if __name__ == "__main__":
    clf_file = f'{BASE_DIR}/output/dis_classifiers.sav'
    extract_features_per_mouse('bsoid_strain_data.csv', clf_file)