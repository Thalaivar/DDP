import os
try:
    import umap
except ModuleNotFoundError:
    pass
import ray
import yaml
import psutil
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
from BSOID.clustering import *
from BSOID.preprocessing import *

from BSOID.features import extract_comb_feats, aggregate_features

from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

class BSOID:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.run_id     = config["run_id"]
        base_dir        = os.path.join(config["base_dir"], self.run_id)
        self.base_dir   = base_dir
        self.raw_dir    = os.path.join(base_dir, "raw")
        self.csv_dir    = os.path.join(base_dir, "csvs")
        self.output_dir = os.path.join(base_dir, "output")

        self.fps            = config["fps"]
        self.stride_window  = round(config["stride_window"] * self.fps / 1000)
        self.conf_threshold = config["conf_threshold"]
        self.bodyparts      = config["bodyparts"]
        self.filter_thresh  = config["filter_thresh"]
        self.trim_params    = config["trim_params"]
        self.num_points     = config["num_points"]
        self.hdbscan_params = config["hdbscan_params"]
        self.umap_params    = config["umap_params"]
        self.jax_dataset    = config["JAX_DATASET"]
        
        self.scale_before_umap = config["scale_before_umap"]
        
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
    
    def load_from_dataset(self, n=None, n_strains=None, min_video_len=120):
        """
        Load raw data files from the JAX database. Files are read in HDF5 format and then 
        confidence thresholded to be saved on disk with the required bodyparts data extracted.

        Inputs:
            - n (int) : number of animals to consider per strain
            - n_strains (int) : number of strains to include in dataset
            - min_video_len (int) : minimum length (in mins) of raw video that the file must contain
        Outputs:
            - filtered_data (dict) : strain-wise filtered keypoint data
        """
        min_video_len *= (60 * self.fps)

        input_csv = self.jax_dataset["input_csv"]
        data_dir = self.jax_dataset["data_dir"]
        filter_thresh = self.filter_thresh

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
                    if conf.shape[0] >= min_video_len:
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
        """
        Get filtered keypoint data and geometric features for a random animal from given strain
        
        Inputs:
            - strain (str) : strain from which random animal is picked
        Outputs:
            - feats : geometric features extracted from animal
            - fdata : fitlered keypoint data extracted from animal
        """
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
        
        feats = extract_comb_feats(fdata, self.fps)
        feats = aggregate_features(feats, self.stride_window)
        return feats, fdata
    def features_from_points(self):
        """
        Extract geometric features from saved dataset (needs to be called only once, and 
        after `load_from_dataset`). Geometric features are extracted for the animals from each strain
        and aggregated into bins.
        """

        filtered_data = self.load_filtered_data()
        logger.info(f'extracting features from {len(filtered_data)} animals')
        
        pbar = tqdm(total=len(filtered_data))
        feats = {}
        for strain, fdata in filtered_data.items():
            feats[strain] = Parallel(n_jobs=psutil.cpu_count(logical=False))(delayed(extract_comb_feats)(data, self.fps) for data in fdata)
            feats[strain] = [aggregate_features(f, self.stride_window) for f in feats[strain]]
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

    def cluster_strainwise(self, logfile=None):
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus)
        
        kwargs = {
            "num_points": self.num_points,
            "umap_params": self.umap_params,
            "hdbscan_params": self.hdbscan_params,
            "scale": self.scale_before_umap,
            "verbose": False
        }

        @ray.remote
        def par_cluster(strain, feats, **kwargs):    
            rep_data, clustering = cluster_for_strain(feats[strain], **kwargs)
            
            if logfile is not None:
                soft_labels = clustering["soft_labels"]
                logger = open(logfile, 'a')
                logger.write(f"Identified {soft_labels.max() + 1} clusters (with entropy ratio: {round(calculate_entropy_ratio(soft_labels), 3)}) from {rep_data.shape} templates for strain: {strain}")

            return strain, rep_data, clustering

        feats = self.load_features(collect=False)
        feats_id = ray.put(feats)

        logger.info(f"processing {len(feats)} strains...")
        futures = [par_cluster.remote(strain, feats_id, **kwargs) for strain in feats.keys()]

        pbar, results = tqdm(total=len(futures)), []
        while len(futures) > 0:
            n = len(futures) if len(futures) < num_cpus else num_cpus
            fin, rest = ray.wait(futures, num_returns=n)
            results.extend(ray.get(fin))
            futures = rest
            pbar.update(n)

        ray.shutdown()

        rep_data, clustering = {}, {}
        for res in results:
            strain, data, labels = res
            rep_data[strain] = data
            clustering[strain] = labels

        with open(os.path.join(self.output_dir, self.run_id + "_strainwise_clustering.sav"), "wb") as f:
            joblib.dump([rep_data, clustering], f)
        
        return rep_data, clustering
    
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

    def load_strainwise_clustering(self):
        with open(self.output_dir + '/' + self.run_id + '_strainwise_clustering.sav', 'rb') as f:
            templates, clustering = joblib.load(f)
        
        return templates, clustering

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