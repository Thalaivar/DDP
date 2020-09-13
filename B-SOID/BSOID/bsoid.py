import os
import random
import joblib
import hdbscan
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from BSOID.clustering import bigCURE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from BSOID.data import download_data, conv_bsoid_format
from BSOID.features import extract_feats, temporal_features, FPS
from BSOID.preprocessing import likelihood_filter, normalize_feats, windowed_feats

class BSOID:
    def __init__(self, run_id: str, 
                base_dir: str, 
                conf_threshold: float=0.3,
                fps: int=30):
        self.conf_threshold = conf_threshold
        self.run_id = run_id
        self.raw_dir = base_dir + '/raw'
        self.csv_dir = base_dir + '/csvs'
        self.output_dir = base_dir + '/output'
        self.fps = fps

        try:
            os.mkdir(self.output_dir)    
        except FileExistsError:
            pass
        try:
            os.mkdir(self.csv_dir)    
        except FileExistsError:
            pass
    
    def get_data(self, n=None, download=False):
        if download:
            download_data('bsoid_strain_data.csv', self.raw_dir)
        
        files = os.listdir(self.raw_dir)
        logging.info("converting {} HDF5 files to csv files".format(len(files)))
        if n is not None:
            files = random.sample(files, n)
        for i in tqdm(range(len(files))):
            if files[i][-3:] == ".h5":
                conv_bsoid_format(self.raw_dir+'/'+files[i], self.csv_dir)

    def process_csvs(self, parallel=True):
        csv_data_files = os.listdir(self.csv_dir)
        csv_data_files = [self.csv_dir + '/' + f for f in csv_data_files]

        logging.info('processing {} csv files from {}'.format(len(csv_data_files), self.csv_dir))

        if not parallel:
            filtered_data = []
            for i in tqdm(range(len(csv_data_files))):
                data = pd.read_csv(csv_data_files[i])
                filtered_data.append(likelihood_filter(data, self.conf_threshold))
        else:
            from joblib import Parallel, delayed
            filtered_data = Parallel(n_jobs=-1, backend="multiprocessing")(
                    delayed(likelihood_filter)(pd.read_csv(data), self.conf_threshold) for data in csv_data_files)
        
        with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'wb') as f:
            joblib.dump(filtered_data, f)

        return filtered_data
    
    def features_from_points(self, temporal_window=16, stride_window=3, temporal_dims=None):
        with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'rb') as f:
            filtered_data = joblib.load(f)
        
        n_animals = len(filtered_data)
        logging.info('extracting features from filtered data of {} animals'.format(n_animals))

        feats = [extract_feats(filtered_data[i]) for i in tqdm(range(n_animals))]

        feats = np.vstack((feats))
        logging.info('extracted {} samples of {}D features'.format(*feats.shape))

        # indices 0-6 are link lengths, during windowing they should be averaged
        win_feats_ll = windowed_feats(feats[:,:7], stride_window, mode='mean')
        # indices 7-13 are relative angles, during windowing they should be summed
        win_feats_rth = windowed_feats(feats[:,7:], stride_window, mode='sum')
        feats = np.hstack((win_feats_ll, win_feats_rth))        
        logging.info('collected features into {}ms bins for dataset shape [{},{}]'.format(stride_window*FPS, *feats.shape))

        feats, temporal_feats = temporal_features(feats, temporal_window)
        if temporal_dims is not None:
            logging.info('reducing {} temporal features dimension from {}D to {}D'.format(*temporal_feats.shape, temporal_dims))
            pca = PCA(n_components=temporal_dims).fit(temporal_feats)
            temporal_feats = pca.transform(temporal_feats)
        feats = np.hstack((feats, temporal_feats))
        logging.info('extracted temporal features for final data set of shape [{},{}]'.format(*feats.shape))

        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'wb') as f:
            joblib.dump([feats, temporal_feats], f)

        return feats, temporal_feats
        
    def cluster_feats(self, min_cluster_prop=0.1, scale_feats=True):
        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'rb') as f:
            feats, _ = joblib.load(f)

        scaler = StandardScaler().fit(feats)
        feats_sc = scaler.transform(feats)        

        min_cluster_size = int(round(min_cluster_prop * 0.01 * feats_sc.shape[0]))
        clusterer = hdbscan.HDBSCAN(min_cluster_size, min_samples=10, prediction_data=True).fit(feats_sc)
        assignments = clusterer.labels_
        soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
        soft_assignments = np.argmax(soft_clusters, axis=1)
        
        logging.info('identified {} clusters from {} samples in {}D'.format(len(np.unique(assignments)), *feats_sc.shape))

        with open(self.output_dir + '/' + self.run_id + '_clusters.sav', 'wb') as f:
            joblib.dump([assignments, soft_clusters, soft_assignments], f)

        return assignments, soft_clusters, soft_assignments
            