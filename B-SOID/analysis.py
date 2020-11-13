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
        # code to retrieve data from the server
        strain, data, movie_name = self.filename.split('/')
        logging.info(f'extracting data for strain {self.strain} from dataset {data} with mouse ID {self.id}')
        
        session = ftplib.FTP("ftp.box.com")
        session.login("ae16b011@smail.iitm.ac.in", "Q0w9e8r7t6Y%Z")

        master_dir = 'JAX-IITM Shared Folder/Datasets/'
        idx = STRAINS.index(strain)
        if idx == 0:
            movie_dir = master_dir + DATASETS[0] + strain + "/" + data + "/"
            session.cwd(movie_dir)
        elif idx == 5:
            movie_dir = master_dir + DATASETS[4] + strain + "/" + data + "/" 
            session.cwd(movie_dir)
        else:
            try:
                movie_dir = master_dir + DATASETS[idx-1] + strain + "/" + data + "/"
                session.cwd(movie_dir)
            except:
                movie_dir = master_dir + DATASETS[idx] + strain + "/" + data + "/"
                session.cwd(movie_dir)
        
        filename = movie_name[0:-4] + "_pose_est_v2.h5"
        session.retrbinary("RETR "+ filename, open(BASE_DIR + '/' + filename, 'wb').write)
        session.close()
        filename = f'{BASE_DIR}/{filename}'
        logging.info(f'temporarily saved downloaded file to {filename}')
        
        f = h5py.File(filename, 'r')
        data = list(f.keys())[0]
        keys = list(f[data].keys())
        conf, pos = np.array(f[data][keys[0]]), np.array(f[data][keys[1]])
        f.close()
        os.remove(filename)

        # trim start and end
        end_trim = FPS*2*60
        conf, pos = conf[end_trim:-end_trim,:], pos[end_trim:-end_trim,:]
        data = _bsoid_format(conf, pos)
        fdata, perc_filt = likelihood_filter(data)

        if perc_filt > 10:
            logging.warn(f'% data filtered from raw data is too high ({perc_filt} %)')
        
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
        if os.path.exists(mouse.save_dir):
            pass
        else:
            mouse.extract_features()
            mouse.save()

    from joblib import Parallel, delayed
    Parallel(n_jobs=-1)(delayed(extract)(i, data) for i in range(N))
        

if __name__ == "__main__":
    clf_file = f'{BASE_DIR}/output/dis_classifiers.sav'
    extract_features_per_mouse('bsoid_strain_data.csv', clf_file)