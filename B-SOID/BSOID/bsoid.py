import os
try:
    import umap
except ModuleNotFoundError:
    pass
import random
import yaml
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

from BSOID.features import extract_comb_feats as extract_feats

from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

class BSOID:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.run_id = config["run_id"]
        base_dir = os.path.join(config["base_dir"], self.run_id)
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.csv_dir = os.path.join(base_dir, "csvs")
        self.output_dir = os.path.join(base_dir, "output")

        self.fps = config["fps"]
        self.stride_window = round(config["stride_window"] * self.fps / 1000)
        self.conf_threshold = config["conf_threshold"]
        self.bodyparts = config["bodyparts"]
        self.filter_thresh = config["filter_thresh"]
        self.trim_params = config["trim_params"]
        self.hdbscan_params = config["hdbscan_params"]
        self.umap_params = config["umap_params"]
        self.mlp_params = config["mlp_params"]
        self.jax_dataset = config["JAX_DATASET"]
        self.reduced_dim = config["reduced_dim"]
        self.sample_size = config["sample_size"]
        self.cluster_range = config["cluster_range"]
        
        for d in [self.base_dir, self.output_dir, self.csv_dir, self.raw_dir]:
            try: os.mkdir(d)
            except FileExistsError: pass
        
        with open(os.path.join(self.base_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, default_flow_style=False)
            
        self.describe()

    def get_data(self, n=None, download=False, parallel=False):
        try:
            os.mkdir(self.csv_dir)    
        except FileExistsError:
            pass

        if download:
            download_data('bsoid_strain_data.csv', self.raw_dir)
        
        files = os.listdir(self.raw_dir)
        logger.info("converting {} HDF5 files to csv files".format(len(files)))
        if n is not None:
            files = random.sample(files, n)
        if parallel:
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1)(delayed(extract_to_csv)(self.raw_dir+'/'+files[i], self.csv_dir) for i in range(len(files)))
        else:
            for i in tqdm(range(len(files))):
                if files[i][-3:] == ".h5":
                    extract_to_csv(self.raw_dir+'/'+files[i], self.csv_dir)

    def process_csvs(self):
        csv_data_files = os.listdir(self.csv_dir)
        csv_data_files = [self.csv_dir + '/' + f for f in csv_data_files if f.endswith('.csv')]

        logger.info('processing {} csv files from {}'.format(len(csv_data_files), self.csv_dir))

        filtered_data = []
        skipped = 0
        for i in range(len(csv_data_files)):
            data = pd.read_csv(csv_data_files[i])
            fdata, perc_filt = likelihood_filter(data, fps=self.fps, conf_threshold=self.conf_threshold, bodyparts=self.bodyparts, **self.trim_params)
            if fdata is not None and perc_filt < self.filter_thresh:
                assert fdata['x'].shape == fdata['y'].shape == fdata['conf'].shape, 'filtered data shape does not match across x, y, and conf values'
                filtered_data.append(fdata)
                shape = fdata['x'].shape
                logger.info(f'preprocessed {shape} data from animal #{i}, with {round(perc_filt, 2)}% data filtered')
            else:
                logger.info(f'skpping {i}-th dataset since % filtered is {round(perc_filt, 2)}')
                skipped += 1

        logger.info(f'skipped {skipped}/{len(csv_data_files)} datasets')
        with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'wb') as f:
            joblib.dump(filtered_data, f)

        return filtered_data
    
    def load_from_dataset(self, n=None, n_strains=None, min_video_length=120):
        input_csv = self.jax_dataset["input_csv"]
        data_dir = self.jax_dataset["data_dir"]
        filter_thresh = self.filter_thresh
        min_video_length *= 60 * 30

        if input_csv.endswith('.tsv'):
            all_data = pd.read_csv(input_csv, sep='\t')    
        else:
            all_data = pd.read_csv(input_csv)
        all_data = list(all_data.groupby("Strain"))

        def filter_for_strain(group_strain, raw_data, n):
            n = raw_data.shape if n is None else n
            
            count, strain_fdata = 0, []
            raw_data = raw_data.sample(frac=1)
            
            for j in range(raw_data.shape[0]):
                if count >= n:
                    break

                metadata = dict(raw_data.iloc[j])
                try:
                    pose_dir, _ = get_pose_data_dir(data_dir, metadata['NetworkFilename'])
                    _, _, movie_name = metadata['NetworkFilename'].split('/')
                    filename = f'{pose_dir}/{movie_name[0:-4]}_pose_est_v2.h5'

                    conf, pos = process_h5py_data(h5py.File(filename, "r"))

                    if conf.shape[0] >= min_video_length:
                        bsoid_data = bsoid_format(conf, pos)
                        fdata, perc_filt = likelihood_filter(bsoid_data, self.fps, self.conf_threshold, bodyparts=self.bodyparts, **self.trim_params)
                        strain, mouse_id = metadata['Strain'], metadata['MouseID']
                        
                        if perc_filt > filter_thresh:
                            logger.warning(f'mouse:{strain}/{mouse_id}: % data filtered from raw data is too high ({perc_filt} %)')
                        else:
                            shape = fdata['x'].shape
                            logger.debug(f'preprocessed {shape} data from {strain}/{mouse_id} with {round(perc_filt, 2)}% data filtered')
                            strain_fdata.append(fdata)
                            count += 1
                except Exception as e:
                    logger.warning(e)
                    pass
            
            logger.info(f"extracted {len(strain_fdata)} animal data from strain {group_strain}")
            return (strain_fdata, group_strain)
        
        logger.info(f"extracting from {len(all_data)} strains with {n} animals per strain")

        import psutil
        num_cpus = psutil.cpu_count(logical=False)    
        filtered_data = Parallel(n_jobs=num_cpus)(delayed(filter_for_strain)(*all_data[i], n) for i in tqdm(range(len(all_data))))
        
        filtered_data = [data for data in filtered_data if len(data[0]) > 0]
        if n_strains is not None:
            filtered_data = random.sample(filtered_data, n_strains)
        
        filtered_data = {strain: data for data, strain in filtered_data}
        
        logger.info(f"extracted data from {len(filtered_data)} strains with a total of {sum(len(data) for _, data in filtered_data.items())} animals")
        with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'wb') as f:
            joblib.dump(filtered_data, f)

        return filtered_data

    def get_random_animal_data(self, strain):
        input_csv = self.jax_dataset["input_csv"]
        data_dir = self.jax_dataset["data_dir"]
        filter_thresh = self.filter_thresh

        if input_csv.endswith('.tsv'):
            data = pd.read_csv(input_csv, sep='\t')    
        else:
            data = pd.read_csv(input_csv)
        data = data.groupby("Strain")
        data = data.get_group(strain)
        data = data.sample(frac=1)

        k = 0
        while True:
            metadata = dict(data.iloc[k])
            try:
                pose_dir, _ = get_pose_data_dir(data_dir, metadata['NetworkFilename'])
                _, _, movie_name = metadata['NetworkFilename'].split('/')
                filename = f'{pose_dir}/{movie_name[0:-4]}_pose_est_v2.h5'

                conf, pos = process_h5py_data(h5py.File(filename, "r"))

                bsoid_data = bsoid_format(conf, pos)
                fdata, perc_filt = likelihood_filter(bsoid_data, self.fps, self.conf_threshold, bodyparts=self.bodyparts, **self.trim_params)
                strain, mouse_id = metadata['Strain'], metadata['MouseID']
                
                if perc_filt > filter_thresh:
                    logger.warning(f'mouse:{strain}/{mouse_id}: % data filtered from raw data is too high ({perc_filt} %)')
                else:
                    shape = fdata['x'].shape
                    logger.debug(f'preprocessed {shape} data from {strain}/{mouse_id} with {round(perc_filt, 2)}% data filtered')
                    break
            except Exception as e:
                logger.warning(e)
                pass
            
            k += 1
        
        feats = extract_feats(fdata, self.fps, self.stride_window)
        return feats, fdata

    def features_from_points(self):
        filtered_data = self.load_filtered_data()
        logger.info(f'extracting features from {len(filtered_data)} animals')
        
        # extract geometric features

        pbar = tqdm(total=len(filtered_data))
        feats = {}
        for strain, fdata in filtered_data.items():
            feats[strain] = Parallel(n_jobs=-1)(delayed(extract_feats)(data, self.fps, self.stride_window) for data in fdata)
            pbar.update(1)

        logger.info(f'extracted {len(feats)} datasets of {feats[list(feats.keys())[0]][0].shape[1]}D features')
        logger.info(f'collected features into bins of {1000 * self.stride_window // self.fps} ms')

        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'wb') as f:
            joblib.dump(feats, f)

    def best_reduced_dim(self, var_prop=0.7):
        _, feats_sc = self.load_features(collect=True)
        pca = PCA().fit(feats_sc)
        num_dimensions = np.argwhere(np.cumsum(pca.explained_variance_ratio_) >= var_prop)[0][0] + 1
        print(f'At least {num_dimensions} dimensions are needed to retain {var_prop} of the total variance')

    def umap_reduce(self):
        reduced_dim, sample_size = self.reduced_dim, self.sample_size        
        feats, feats_sc = self.load_features(collect=True)

        if sample_size > 1:
            idx = np.random.permutation(np.arange(feats.shape[0]))[0:sample_size]
            feats_train = feats_sc[idx,:]
            feats_usc = feats[idx, :]
        else:
            feats_train = feats_sc
            feats_usc = feats

        logger.info('running UMAP on {} samples from {}D to {}D with params: {}'.format(*feats_train.shape, reduced_dim, self.umap_params))
        mapper = umap.UMAP(n_components=reduced_dim,  **self.umap_params).fit(feats_train)

        with open(self.output_dir + '/' + self.run_id + '_umap.sav', 'wb') as f:
            joblib.dump([feats_usc, feats_train, mapper.embedding_, mapper], f)
        
        return [feats_usc, feats_train, mapper.embedding_, mapper]

    def identify_clusters_from_umap(self):
        cluster_range = self.cluster_range
        with open(self.output_dir + '/' + self.run_id + '_umap.sav', 'rb') as f:
            _, _, umap_embeddings, _ = joblib.load(f)

        logger.info(f'clustering {umap_embeddings.shape[0]} in {umap_embeddings.shape[1]}D with cluster range={cluster_range}')
        assignments, soft_clusters, soft_assignments, best_clf = cluster_with_hdbscan(umap_embeddings, cluster_range, self.hdbscan_params)
        logger.info('identified {} clusters from {} samples in {}D'.format(len(np.unique(soft_assignments)), *umap_embeddings.shape))

        with open(self.output_dir + '/' + self.run_id + '_clusters.sav', 'wb') as f:
            joblib.dump([assignments, soft_clusters, soft_assignments, best_clf], f)

        return assignments, soft_clusters, soft_assignments, best_clf

    def train_classifier(self):
        _, _, soft_assignments, _ = self.load_identified_clusters()
        _, feats_sc, _ = self.load_umap_results(collect=True)

        logger.info('training neural network on {} scaled samples in {}D'.format(*feats_sc.shape))
        clf = MLPClassifier(**self.mlp_params).fit(feats_sc, soft_assignments)

        with open(self.output_dir + '/' + self.run_id + '_classifiers.sav', 'wb') as f:
            joblib.dump(clf, f)

    def validate_classifier(self):
        _, _, soft_assignments, _ = self.load_identified_clusters()
        _, feats_sc, _ = self.load_umap_results(collect=True)

        logger.info('validating classifier on {} features'.format(*feats_sc.shape))
        feats_train, feats_test, labels_train, labels_test = train_test_split(feats_sc, soft_assignments)
        clf = MLPClassifier(**self.mlp_params).fit(feats_train, labels_train)
        sc_scores = cross_val_score(clf, feats_test, labels_test, cv=5, n_jobs=-1)
        sc_cf = create_confusion_matrix(feats_test, labels_test, clf)
        logger.info('classifier accuracy: {} +- {}'.format(sc_scores.mean(), sc_scores.std())) 
         
        with open(self.output_dir + '/' + self.run_id + '_validation.sav', 'wb') as f:
            joblib.dump([sc_scores, sc_cf], f)

    def create_examples(self, csv_dir, vid_dir, bout_length=3, n_examples=10):
        csv_files = [csv_dir + '/' + f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        video_files = [vid_dir + '/' + f for f in os.listdir(vid_dir) if f.endswith('.avi')]

        csv_files.sort()
        video_files.sort()

        n_animals = len(csv_files)
        logger.info(f'generating {n_examples} examples from {n_animals} videos each with minimum bout length of {1000 * bout_length / self.fps} ms')

        labels = []
        frame_dirs = []
        for i in range(n_animals):
            label, frame_dir = self.label_frames(csv_files[i], video_files[i])
            labels.append(label)
            frame_dirs.append(frame_dir)
        
        output_path = os.path.join(self.base_dir, "results")
        try:
            os.mkdir(output_path)
        except FileExistsError:
            logger.info(f'results directory: {output_path} already exists, deleting')
            [os.remove(output_path+'/'+f) for f in os.listdir(output_path)]

        clip_window = self.trim_params['end_trim']*60*self.fps
        collect_all_examples(labels, frame_dirs, output_path, bout_length, n_examples, self.fps, clip_window)

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
            frames_from_video(video_file, frame_dir)
        
        logger.debug('extracting features from {}'.format(csv_file))
        
        # filter data from test file
        data = pd.read_csv(csv_file, low_memory=False)
        data, _ = likelihood_filter(data, self.fps, self.conf_threshold, end_trim=self.trim_params['end_trim'], clip_window=0)

        feats = frameshift_features(data, self.stride_window, self.fps, extract_feats, window_extracted_feats)

        with open(self.output_dir + '/' + self.run_id + '_classifiers.sav', 'rb') as f:
            clf = joblib.load(f)

        labels = frameshift_predict(feats, clf, self.stride_window)
        logger.info(f'predicted {len(labels)} frames in {feats[0].shape[1]}D with trained classifier')

        return labels, frame_dir
    
    def load_filtered_data(self):
        with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'rb') as f:
            filtered_data = joblib.load(f)
        
        return filtered_data

    def load_features(self, collect):        
        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'rb') as f:
            feats = joblib.load(f)
        if collect:
            feats_ = []
            for _, data in feats.items():
                feats_.extend(data)
            feats = feats_
            del feats_

            feats_sc = [StandardScaler().fit_transform(data) for  data in feats]
            feats, feats_sc = np.vstack(feats), np.vstack(feats_sc)
            return feats, feats_sc
        else:
            return feats

    def load_identified_clusters(self):
        with open(self.output_dir + '/' + self.run_id + '_clusters.sav', 'rb') as f:
            assignments, soft_clusters, soft_assignments, best_clf = joblib.load(f)
        
        return assignments, soft_clusters, soft_assignments, best_clf

    def load_classifier(self):
        with open(self.output_dir + '/' + self.run_id + '_classifiers.sav', 'rb') as f:
            clf = joblib.load(f)
        
        return clf

    def load_umap_results(self, collect=None):
        with open(self.output_dir + '/' + self.run_id + '_umap.sav', 'rb') as f:
            feats_usc, feats_sc, umap_embeddings = joblib.load(f)
        return feats_usc, feats_sc, umap_embeddings

    def save(self):
        with open(self.output_dir + '/' + self.run_id + '_bsoid.model', 'wb') as f:
            joblib.dump(self, f)

    def describe(self):
        s = (f'    Run ID       : {self.run_id}\n'
             f' Save Location   : {self.base_dir}/output\n'
             f'      FPS        : {self.fps}\n'
             f' Min. Confidence : {self.conf_threshold}\n'
             f'  Stride Window  : {self.stride_window * 1000 // self.fps}ms\n')
        print(s)