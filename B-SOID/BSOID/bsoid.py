import os
try:
    import umap
except ModuleNotFoundError:
    pass
import random
import joblib
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from psutil import virtual_memory
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from BSOID.utils import *
from BSOID.data import *
from BSOID.preprocessing import *
from BSOID.prediction import *

from BSOID.features.displacement_feats import *
# from BSOID.features.bsoid_features import *

MLP_PARAMS = {
    'hidden_layer_sizes': (100, 10),  # 100 units, 10 layers
    'activation': 'logistic',  # logistics appears to outperform tanh and relu
    'solver': 'adam',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,  # learning rate not too high
    'alpha': 0.0001,  # regularization default is better than higher values.
    'max_iter': 1000,
    'early_stopping': False,
    'verbose': 1  # set to 1 for tuning your feedforward neural network
}

UMAP_PARAMS = {
    'min_dist': 0.0,  # small value
    'n_neighbors': 300
}

HDBSCAN_PARAMS = {
    'min_samples': 10,
    'prediction_data': True,
}

TRIM_PARAMS = {
    'end_trim': 2,
    'clip_window': 30
}

class BSOID:
    def __init__(self, run_id: str, 
                base_dir: str, 
                fps: int=30, 
                stride_window: int=3,
                conf_threshold: float=0.3):
        self.run_id = run_id
        self.base_dir = base_dir
        self.raw_dir = base_dir + '/raw'
        self.csv_dir = base_dir + '/csvs'
        self.output_dir = base_dir + '/output'
        self.test_dir = base_dir + '/test'

        # frame rate for video
        self.fps = fps
        # feature extraction parameters
        self.stride_window = stride_window
        self.conf_threshold = conf_threshold

        try:
            os.mkdir(self.output_dir)    
        except FileExistsError:
            pass
        try:
            os.mkdir(self.test_dir)
        except FileExistsError:
            pass

    def get_data(self, n=None, download=False, parallel=False):
        try:
            os.mkdir(self.csv_dir)    
        except FileExistsError:
            pass

        if download:
            download_data('bsoid_strain_data.csv', self.raw_dir)
        
        files = os.listdir(self.raw_dir)
        logging.info("converting {} HDF5 files to csv files".format(len(files)))
        if n is not None:
            files = random.sample(files, n)
        if parallel:
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1)(delayed(extract_to_csv)(self.raw_dir+'/'+files[i], self.csv_dir) for i in range(len(files)))
        else:
            for i in tqdm(range(len(files))):
                if files[i][-3:] == ".h5":
                    extract_to_csv(self.raw_dir+'/'+files[i], self.csv_dir)

    def process_csvs(self, filter_thresh=5):
        csv_data_files = os.listdir(self.csv_dir)
        csv_data_files = [self.csv_dir + '/' + f for f in csv_data_files if f.endswith('.csv')]

        logging.info('processing {} csv files from {}'.format(len(csv_data_files), self.csv_dir))

        filtered_data = []
        skipped = 0
        for i in range(len(csv_data_files)):
            data = pd.read_csv(csv_data_files[i])
            fdata, perc_filt = likelihood_filter(data, fps=self.fps, conf_threshold=self.conf_threshold,**TRIM_PARAMS)
            if fdata is not None and perc_filt < filter_thresh:
                assert fdata['x'].shape == fdata['y'].shape == fdata['conf'].shape, 'filtered data shape does not match across x, y, and conf values'
                filtered_data.append(fdata)
                logging.info(f'preprocessed {fdata['x'].shape} data from animal #{i}, with {perc_filt}% data filtered')
            else:
                logging.info(f'skpping {i}-th dataset since % filtered is {round(perc_filt, 2)}')
                skipped += 1

        logging.info(f'skipped {skipped}/{len(csv_data_files)} datasets')
        with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'wb') as f:
            joblib.dump(filtered_data, f)

        return filtered_data
    
    def features_from_points(self, parallel=False):
        filtered_data = self.load_filtered_data()
        
        # extract geometric features
        if parallel:
            from joblib import Parallel, delayed
            feats = Parallel(n_jobs=2)(delayed(extract_feats)(data, self.fps) for data in filtered_data)
        else:
            feats = []
            for i in tqdm(range(len(filtered_data))):
                feats.append(extract_feats(filtered_data[i], self.fps))

        logging.info(f'extracted {len(feats)} datasets of {feats[0].shape[1]}D features')

        # feats = window_extracted_feats(feats, self.stride_window, self.temporal_window, self.temporal_dims)
        feats = window_extracted_feats(feats, self.stride_window, self.temporal_window, self.temporal_dims)
        logging.info(f'collected features into bins of {1000 * self.stride_window // self.fps} ms')
        
        if self.temporal_window is not None:
            logging.info('combined temporal features to get {} datasets of {}D'.format(len(feats), feats[0].shape[1]))

        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'wb') as f:
            joblib.dump(feats, f)

    def best_reduced_dim(self, var_prop=0.7):
        _, feats_sc = self.load_features()
        pca = PCA().fit(feats_sc)
        num_dimensions = np.argwhere(np.cumsum(pca.explained_variance_ratio_) >= var_prop)[0][0] + 1
        print(f'At least {num_dimensions} dimensions are needed to retain {var_prop} of the total variance')

    def umap_reduce(self, reduced_dim, sample_size=int(5e5)):        
        feats, feats_sc = self.load_features()

        if sample_size > 1:
            idx = np.random.permutation(np.arange(feats.shape[0]))[0:sample_size]
            feats_train = feats_sc[idx,:]
            feats_usc = feats[idx, :]
        else:
            feats_train = feats_sc
            feats_usc = feats

        logging.info('running UMAP on {} samples from {}D to {}D'.format(*feats_train.shape, reduced_dim))
        mapper = umap.UMAP(n_components=reduced_dim,  **UMAP_PARAMS).fit(feats_train)

        with open(self.output_dir + '/' + self.run_id + '_umap.sav', 'wb') as f:
            joblib.dump([feats_usc, feats_train, mapper.embedding_], f)
        
        return [feats_usc, feats_train, mapper.embedding_]

    def identify_clusters_from_umap(self, cluster_range=[0.4,1.2]):
        with open(self.output_dir + '/' + self.run_id + '_umap.sav', 'rb') as f:
            _, _, umap_embeddings = joblib.load(f)

        logging.info(f'clustering {umap_embeddings.shape[0]} in {umap_embeddings.shape[1]}D with cluster range={cluster_range}')
        assignments, soft_clusters, soft_assignments, best_clf = cluster_with_hdbscan(umap_embeddings, cluster_range, HDBSCAN_PARAMS)
        logging.info('identified {} clusters from {} samples in {}D'.format(len(np.unique(soft_assignments)), *umap_embeddings.shape))

        with open(self.output_dir + '/' + self.run_id + '_clusters.sav', 'wb') as f:
            joblib.dump([assignments, soft_clusters, soft_assignments, best_clf], f)

        return assignments, soft_clusters, soft_assignments, best_clf

    def train_classifier(self):
        _, _, soft_assignments, _ = self.load_identified_clusters()
        _, feats_sc, _ = self.load_umap_results(collect=True)

        logging.info('training neural network on {} scaled samples in {}D'.format(*feats_sc.shape))
        clf = MLPClassifier(**MLP_PARAMS).fit(feats_sc, soft_assignments)

        with open(self.output_dir + '/' + self.run_id + '_classifiers.sav', 'wb') as f:
            joblib.dump(clf, f)

    def validate_classifier(self):
        _, _, soft_assignments, _ = self.load_identified_clusters()
        _, feats_sc, _ = self.load_umap_results(collect=True)

        logging.info('validating classifier on {} features'.format(*feats_sc.shape))
        feats_train, feats_test, labels_train, labels_test = train_test_split(feats_sc, soft_assignments)
        clf = MLPClassifier(**MLP_PARAMS).fit(feats_train, labels_train)
        sc_scores = cross_val_score(clf, feats_test, labels_test, cv=5, n_jobs=-1)
        sc_cf = create_confusion_matrix(feats_test, labels_test, clf)
        logging.info('classifier accuracy: {} +- {}'.format(sc_scores.mean(), sc_scores.std())) 
         
        with open(self.output_dir + '/' + self.run_id + '_validation.sav', 'wb') as f:
            joblib.dump([sc_scores, sc_cf], f)

    def create_examples(self, csv_dir, vid_dir, bout_length=3, n_examples=10):
        csv_files = [csv_dir + '/' + f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        video_files = [vid_dir + '/' + f for f in os.listdir(vid_dir) if f.endswith('.avi')]

        csv_files.sort()
        video_files.sort()

        n_animals = len(csv_files)
        logging.info(f'generating {n_examples} examples from {n_animals} videos each with minimum bout length of {1000 * bout_length / self.fps} ms')

        labels = []
        frame_dirs = []
        for i in range(n_animals):
            label, frame_dir = self.label_frames(csv_files[i], video_files[i])
            labels.append(label)
            frame_dirs.append(frame_dir)
        
        output_path = self.base_dir + '/' + self.run_id + '_results'
        try:
            os.mkdir(output_path)
        except FileExistsError:
            logging.info(f'results directory: {output_path} already exists, deleting')
            [os.remove(output_path+'/'+f) for f in os.listdir(output_path)]

        clip_len = None
        if self.temporal_window is not None:
            clip_len = (self.temporal_window - self.stride_window) // 2
        collect_all_examples(labels, frame_dirs, output_path, clip_len, bout_length, n_examples, self.fps)

    def label_frames(self, csv_file, video_file):
        # directory to store results for video
        output_dir = self.test_dir + '/' + csv_file.split('/')[-1][:-4]
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        
        frame_dir = output_dir + '/pngs'
        extract_frames = True
        try:
            os.mkdir(frame_dir)
        except FileExistsError:
            extract_frames = False
        
        # extract 
        if extract_frames:
            logging.info('extracting frames from video {} to dir {}'.format(video_file, frame_dir))
            frames_from_video(video_file, frame_dir)
        
        logging.debug('extracting features from {}'.format(csv_file))
        
        # filter data from test file
        data = pd.read_csv(csv_file, low_memory=False)
        data, _ = likelihood_filter(data, self.fps, end_trim=2)

        feats = frameshift_features(data, self.stride_window, self.fps, extract_feats, window_extracted_feats, self.temporal_window, self.temporal_dims)

        with open(self.output_dir + '/' + self.run_id + '_classifiers.sav', 'rb') as f:
            clf = joblib.load(f)

        labels = frameshift_predict(feats, clf, self.stride_window)
        logging.info(f'predicted {len(labels)} frames in {feats[0].shape[1]}D with trained classifier')

        return labels, frame_dir
    
    def load_filtered_data(self):
        with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'rb') as f:
            filtered_data = joblib.load(f)
        
        return filtered_data

    def load_features(self, collect=True):        
        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'rb') as f:
            feats = joblib.load(f)
        if collect:
            feats_sc = [StandardScaler().fit_transform(data) for data in feats]
            feats, feats_sc = np.vstack(feats), np.vstack(feats_sc)
            return feats, feats_sc
        else:
            return feats

    def load_identified_clusters(self):
        with open(self.output_dir + '/' + self.run_id + '_clusters.sav', 'rb') as f:
            assignments, soft_clusters, soft_assignments, best_clf = joblib.load(f)
        
        return assignments, soft_clusters, soft_assignments, best_clf

    def load_umap_results(self, collect=None):
        with open(self.output_dir + '/' + self.run_id + '_umap.sav', 'rb') as f:
            feats_usc, feats_sc, umap_embeddings = joblib.load(f)
        return feats_usc, feats_sc, umap_embeddings

    def save(self):
        with open(self.output_dir + '/' + self.run_id + '_bsoid.model', 'wb') as f:
            joblib.dump(self, f)

    @staticmethod
    def load_config(base_dir, run_id):
        with open(base_dir + '/output/' + run_id + '_bsoid.model', 'rb') as f:
            config = joblib.load(f)
        
        config.base_dir = base_dir
        config.raw_dir = base_dir + '/raw'
        config.csv_dir = base_dir + '/csvs'
        config.output_dir = base_dir + '/output'
        config.test_dir = base_dir + '/test'
        
        return config