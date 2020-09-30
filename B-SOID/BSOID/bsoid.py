import os
import umap
import random
import joblib
import hdbscan
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from psutil import virtual_memory
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from BSOID.utils import *
from BSOID.data import *
from BSOID.features import *
from BSOID.preprocessing import *
from BSOID.prediction import *
from BSOID.clustering import *

MLP_PARAMS = {
    'hidden_layer_sizes': (100, 10),  # 100 units, 10 layers
    'activation': 'logistic',  # logistics appears to outperform tanh and relu
    'solver': 'adam',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,  # learning rate not too high
    'alpha': 0.0001,  # regularization default is better than higher values.
    'max_iter': 1000,
    'early_stopping': False,
    'verbose': 0  # set to 1 for tuning your feedforward neural network
}

class BSOID:
    def __init__(self, run_id: str, 
                base_dir: str, 
                conf_threshold: float=0.3,
                fps: int=30, 
                temporal_window: int=16,
                stride_window: int=3,
                temporal_dims: int=6):
        self.run_id = run_id
        self.raw_dir = base_dir + '/raw'
        self.csv_dir = base_dir + '/csvs'
        self.output_dir = base_dir + '/output'
        self.test_dir = base_dir + '/test'

        # frame rate for video
        self.fps = fps
        # feature extraction parameters
        self.conf_threshold = conf_threshold
        self.temporal_dims = temporal_dims
        self.temporal_window = temporal_window
        self.stride_window = stride_window

        try:
            os.mkdir(self.output_dir)    
        except FileExistsError:
            pass
        try:
            os.mkdir(self.csv_dir)    
        except FileExistsError:
            pass
        try:
            os.mkdir(self.test_dir)
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
    
    def features_from_points(self):
        filtered_data = self.load_filtered_data()
        
        feats = extract_geo_feats(filtered_data)

        pca = None
        temporal_feats = None
        if self.temporal_window is not None:
            feats, temporal_feats = temporal_features(feats, self.temporal_window)
        
            if self.temporal_dims is not None:
                logging.info('reducing {} temporal features dimension from {}D to {}D'.format(*temporal_feats.shape, self.temporal_dims))
                pca = PCA(n_components=self.temporal_dims).fit(temporal_feats)
                temporal_feats = pca.transform(temporal_feats)          
            
            feats = np.hstack((feats, temporal_feats))
            logging.info('extracted temporal features for final data set of shape [{},{}]'.format(*feats.shape))
        
        feats = window_extracted_feats(feats, self.stride_window)

        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'wb') as f:
            joblib.dump([feats, temporal_feats, pca], f)

        return feats, temporal_feats
        
    def bsoid_features(self):
        filtered_data = self.load_filtered_data()

        feats = extract_bsoid_feats(filtered_data)

        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'wb') as f:
            joblib.dump(feats, f)

    def umap_reduce(self, reduced_dim=10, sample_size=int(5e5), shuffle=True):
        feats, _, _ = self.load_features()
        
        feats_train = StandardScaler().fit_transform(feats)

        # take subset of data
        if shuffle:
            idx = np.random.permutation(np.arange(feats.shape[0]))
        else:
            idx = np.arange(feats.shape[0])
        feats_train = feats_train[idx[:sample_size],:]
        feats_usc = feats[idx[:sample_size], :]
        
        logging.info('running UMAP on {} samples from {}D to {}D'.format(*feats_train.shape, reduced_dim))
        mapper = umap.UMAP(n_components=reduced_dim, n_neighbors=100, min_dist=0.0).fit(feats_train)
        
        with open(self.output_dir + '/' + self.run_id + '_umap.sav', 'wb') as f:
            joblib.dump([feats_usc, feats_train, mapper.embedding_], f)


    def umap_reduce_all(self, reduced_dim=10, sample_size=int(5e5), shuffle=True):
        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'rb') as f:
            feats, _ = joblib.load(f)

        # take subset of data
        if shuffle:
            idx = np.random.permutation(np.arange(feats.shape[0]))
        else:
            idx = np.arange(feats.shape[0])
        feats_train = feats[idx[:sample_size],:]
        rem_feats = feats[idx[sample_size:],:]
        
        logging.info('divided data into {} and {} samples'.format(feats_train.shape[0], rem_feats.shape[0]))

        # scale both datasets
        rem_feats = StandardScaler().fit_transform(rem_feats)
        feats_train = StandardScaler().fit_transform(feats_train)
        
        
        logging.info('running UMAP on {} samples from {}D to {}D'.format(*feats_train.shape, reduced_dim))
        mapper = umap.UMAP(n_components=reduced_dim, n_neighbors=100, min_dist=0.0).fit(feats_train)
        
        with open(self.output_dir + '/' + self.run_id + '_umap_all.sav', 'wb') as f:
            joblib.dump([mapper, mapper.embedding_], f)

        # batch transform rest of data
        embeddings = [mapper.embedding_]
        pbar = tqdm(total=rem_feats.shape[0])
        idx = 0
        batch_sz = sample_size // 10
        logging.info('embedding {} samples from data set to {}D in batches of {}'.format(rem_feats.shape[0], reduced_dim, batch_sz))
        while idx < rem_feats.shape[0]:
            if idx + batch_sz >= rem_feats.shape[0]:
                batch_embed = mapper.transform(rem_feats[idx:,:])
                pbar.update(rem_feats.shape[0]-idx)
            else:
                batch_embed = mapper.transform(rem_feats[idx:idx+batch_sz,:])
                pbar.update(batch_sz)
            embeddings.append(batch_embed)
            idx += batch_sz

        embeddings = np.vstack(embeddings)
        logging.info('finished transforming {} samples'.format(idx))

        with open(self.output_dir + '/' + self.run_id + '_umap_all.sav', 'wb') as f:
            joblib.dump(embeddings, f)

    def max_samples_for_umap(self):
        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'rb') as f:
            feats, _ = joblib.load(f)

        mem = virtual_memory()
        allowed_n = int((mem.available - 256000000)/(feats.shape[1]*32*100))
        
        logging.info('max allowed samples for umap: {} and data has: {}'.format(allowed_n, feats.shape[0]))
        return allowed_n

    def cluster_everything(self):
        feats, _, _= self.load_features()
        feats = StandardScaler().fit_transform(feats)
        
        partitions, assignments = preclustering(feats, n_parts=3, min_clusters=75, max_clusters=100)
        with open(self.output_dir + '/' + self.run_id + '_clusters_all.sav', 'wb') as f:
            joblib.dump([partitions, assignments], f)

        clusters = clusters_from_assignments(partitions, assignments, n_rep=1000, alpha=0.5)

        with open(self.output_dir + '/' + self.run_id + '_clusters_all.sav', 'wb') as f:
            joblib.dump(clusters, f)

    def identify_clusters_from_umap(self, min_cluster_prop=0.1):
        # umap embeddings are to be used directly
        with open(self.output_dir + '/' + self.run_id + '_umap.sav', 'rb') as f:
            _, _, feats_sc = joblib.load(f)

        min_cluster_size = int(round(min_cluster_prop * 0.01 * feats_sc.shape[0]))
        logging.info('clustering {} samples in {}D with HDBSCAN for a minimum cluster size of {}'.format(*feats_sc.shape, min_cluster_size))
        clusterer = hdbscan.HDBSCAN(min_cluster_size, min_samples=10, prediction_data=True).fit(feats_sc)
        assignments = clusterer.labels_
        soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
        soft_assignments = np.argmax(soft_clusters, axis=1)
        
        logging.info('identified {} clusters from {} samples in {}D'.format(len(np.unique(assignments)), *feats_sc.shape))

        with open(self.output_dir + '/' + self.run_id + '_clusters.sav', 'wb') as f:
            joblib.dump([assignments, soft_clusters, soft_assignments], f)

        return assignments, soft_clusters, soft_assignments

    def train_classifier(self):
        with open(self.output_dir + '/' + self.run_id + '_clusters.sav', 'rb') as f:
            _, _, soft_assignments = joblib.load(f)

        with open(self.output_dir + '/' + self.run_id + '_umap.sav', 'rb') as f:
                _, feats, _ = joblib.load(f)

        # try using scaled features to train also
        # feats_sc = StandardScaler().fit_transform(feats)
        feats_sc = feats

        logging.info('training neural network on {} scaled samples in {}D'.format(*feats_sc.shape))
        clf = MLPClassifier(**MLP_PARAMS).fit(feats_sc, soft_assignments)

        with open(self.output_dir + '/' + self.run_id + '_classifiers.sav', 'wb') as f:
            joblib.dump(clf, f)

    def validate_classifier(self):
        with open(self.output_dir + '/' + self.run_id + '_clusters.sav', 'rb') as f:
            _, _, soft_assignments = joblib.load(f)

        with open(self.output_dir + '/' + self.run_id + '_umap.sav', 'rb') as f:
                _, feats_sc, _ = joblib.load(f)
        
        # check with/without scaling
        # logging.info('validating classifier on {} features'.format(*feats_usc.shape))
        # feats_train, feats_test, labels_train, labels_test = train_test_split(feats_usc, soft_assignments)
        # clf = MLPClassifier(**MLP_PARAMS).fit(feats_train, labels_train)
        # usc_scores = cross_val_score(clf, feats_test, labels_test, cv=5, n_jobs=-1)
        # usc_cf = create_confusion_matrix(feats_test, labels_test, clf)
        # logging.info('classifier accuracy: {} +- {}'.format(usc_scores.mean(), usc_scores.std())) 

        logging.info('validating classifier on {} features'.format(*feats_sc.shape))
        feats_train, feats_test, labels_train, labels_test = train_test_split(feats_sc, soft_assignments)
        clf = MLPClassifier(**MLP_PARAMS).fit(feats_train, labels_train)
        sc_scores = cross_val_score(clf, feats_test, labels_test, cv=5, n_jobs=-1)
        sc_cf = create_confusion_matrix(feats_test, labels_test, clf)
        logging.info('classifier accuracy: {} +- {}'.format(sc_scores.mean(), sc_scores.std())) 
         
        with open(self.output_dir + '/' + self.run_id + '_validation.sav', 'wb') as f:
            joblib.dump([sc_scores, sc_cf], f)

    def label_frames(self, csv_file, video_file, extract_frames=True, load_feats=False, **video_args):
        # directory to store results for video
        output_dir = self.test_dir + '/' + csv_file.split('/')[-1][:-4]
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        frame_dir = output_dir + '/pngs'
        try:
            os.mkdir(frame_dir)
        except FileExistsError:
            pass
        
        # extract 
        if extract_frames:
            logging.info('extracting frames from video {} to dir {}'.format(video_file, frame_dir))
            frames_from_video(video_file, frame_dir)
        
        shortvid_dir = output_dir + '/mp4s'
        try:
            os.mkdir(shortvid_dir)
        except FileExistsError:
            pass

        logging.info('saving example videos from {} to {}'.format(video_file, shortvid_dir))
        logging.info('generating {} examples with minimum bout: {} ms'.format(video_args['n_examples'], video_args['bout_length']))

        if load_feats:
            with open(output_dir + '/feats.sav', 'rb') as f:
                feats = joblib.load(f)
        else:
            logging.debug('extracting features from {}'.format(csv_file))
            
            # filter data from test file
            data = pd.read_csv(csv_file, low_memory=False)
            data = [likelihood_filter(data, self.conf_threshold)]

            feats = frameshift_features(data, self.stride_window, self.temporal_window, self.temporal_dims, self.fps)

            with open(output_dir + '/feats.sav', 'wb') as f:
                joblib.dump(feats, f)

        with open(self.output_dir + '/' + self.run_id + '_classifiers.sav', 'rb') as f:
            clf = joblib.load(f)
        
        labels = frameshift_predict(feats, clf, self.stride_window)
        logging.info(f'predicted {len(labels)} frames with trained classifier')

        create_vids(labels, frame_dir, shortvid_dir, self.temporal_window, **video_args)
    
    def load_filtered_data(self):
        with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'rb') as f:
            filtered_data = joblib.load(f)
        
        return filtered_data

    def load_features(self):
        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'rb') as f:
            feats, temporal_feats, pca = joblib.load(f)
        
        return feats, temporal_feats, pca

    def load_identified_clusters(self):
        with open(self.output_dir + '/' + self.run_id + '_clusters.sav', 'rb') as f:
            assignments, soft_clusters, soft_assignments = joblib.load(f)
        
        return assignments, soft_clusters, soft_assignments


    def save(self):
        with open(self.output_dir + '/' + self.run_id + '_bsoid.model', 'wb') as f:
            joblib.dump(self, f)

    @staticmethod
    def load_config(base_dir, run_id):
        with open(base_dir + '/output/' + run_id + '_bsoid.model', 'rb') as f:
            config = joblib.load(f)
        
        config.raw_dir = base_dir + '/raw'
        config.csv_dir = base_dir + '/csvs'
        config.output_dir = base_dir + '/output'
        config.test_dir = base_dir + '/test'
        
        return config)