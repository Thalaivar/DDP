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
    def __init__(self, bsoid_clf_file, window_feats=True, **metadata):
        self.strain = metadata['strain']
        self.filename = metadata['network_filename']
        self.sex = metadata['sex']
        self.id = metadata['mouse_id']

        self.window_feats = window_feats
        with open(bsoid_clf_file, 'rb') as f:
            self.clf = joblib.load(f)

        self.save_dir = f'{MICE_DIR}/{self.strain}/{self.id}'
        try:
            os.makedirs(self.save_dir)
        except:
            pass

    def bigram_probability_matrix(self):
        labels = self.get_behaviour_labels()
        
        n_lab = labels.max() + 1
        bigram_mat = np.zeros((n_lab, n_lab))
        curr_lab = labels[0]
        for i in range(1, labels.size):
            lab = labels[i]
            bigram_mat[curr_lab, lab] += 1
            curr_lab = lab
        
        for i in range(n_lab):
            bigram_mat[i] /= bigram_mat[i].sum()    
        
        np.save(f'{self.save_dir}/bigram_matrix.npy', bigram_mat)

        return bigram_mat

    def get_behaviour_info(self, labels, behaviour_idx, min_bout_len):
        if not isinstance(behaviour_idx, list):
            behaviour_idx = [behaviour_idx]
        
        labels = self.get_behaviour_labels()

        counts = [0 for _ in range(labels.max() + 1)]
        for lab in labels:
            counts[lab] += 1
        
        # total bout duration in entire video
        bin_length = STRIDE_WINDOW / FPS
        total_duration = [counts[idx] * bin_length for idx in behaviour_idx]

        # no. of bouts and their lengths
        n_bouts = [0 for _ in range(labels.max() + 1)]
        bout_lengths = [[] for _ in range(labels.max() + 1)]

        start_lab = labels[0]
        i = 1
        while i < len(labels):
            bout_len = 0
            while start_lab == labels[i]:
                bout_len += 1
                i += 1
            bout_lengths[start_lab].append(bout_len)
            if bout_len > min_bout_len:
                n_bouts[start_lab] += 1
            start_lab = labels[i]
        
        bout_lengths = [sum(lens)/len(lens) for lens in bout_lengths]

        return total_duration

    def get_behaviour_labels(self):
        if os.path.exists(f'{self.save_dir}/feats.npy'):
            feats = self.load_features()
        else:
            feats = self.extract_features()
        labels = self.clf.predict(feats)
        return labels
        
    def extract_features(self, window=True, features_type='dis'):
        data = self._get_raw_data()
        if features_type == 'dis':
            from BSOID.features.displacement_feats import extract_feats, window_extracted_feats
        
        feats = extract_feats(data, FPS)
        if self.window_feats:
            feats = window_extracted_feats([feats], STRIDE_WINDOW, TEMPORAL_WINDOW, TEMPORAL_DIMS)[0]
        logging.info(f'extracted features of shape {feats.shape}')

        np.save(f'{self.save_dir}/feats.npy', feats)
        return feats

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
        with open(f'{self.save_dir}/mouse.pkl', 'wb') as f:
            joblib.dump(self, f)
    
    def load_features(self):
        return np.load(f'{self.save_dir}/feats.npy')

def extract_features_per_mouse(data_lookup_file, clf_file):
    import pandas as pd
    data = pd.read_csv(data_lookup_file)
    N = data.shape[0]

    logging.info(f'extracting raw data for {N} mice')

    def extract(i, data):
        mouse_data = {'strain': data.loc[i]['Strain'].replace('/', '-'),
                'network_filename': data.loc[i]['NetworkFilename'],
                'sex': data.loc[i]['Sex'],
                'mouse_id': data.loc[i]['MouseID']
            }
        mouse = Mouse(**mouse_data, bsoid_clf_file=clf_file)
        if os.path.exists(f'{mouse.save_dir}/feats.npy'):
            pass
        else:
            mouse.extract_features()
            mouse.save()

    from joblib import Parallel, delayed
    Parallel(n_jobs=-1)(delayed(extract)(i, data) for i in range(N))
        

if __name__ == "__main__":
    clf_file = f'{BASE_DIR}/output/dis_classifiers.sav'
    extract_features_per_mouse('bsoid_strain_data.csv', clf_file)