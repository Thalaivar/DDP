import os
import random
import joblib
import logging
import pandas as pd
from tqdm import tqdm
from preprocessing import likelihood_filter, extract_bsoid_feats, normalize_feats
from data import download_data, conv_bsoid_format
from clustering import bigCURE

class BSOID:
    def __init__(self, run_id: str, 
                base_dir: str, 
                conf_threshold: float=0.3,
                fps: int=30):
        self.conf_threshold = conf_threshold
        self.run_id = run_id
        self.raw_dir = base_dir + '/raw'
        self.csv_dir = base_dir + '/csv'
        self.output_dir = base_dir + '/output'
        self.fps = fps

        try:
            os.mkdir(self.output_dir)    
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

    def process_csvs(self):
        csv_data_files = os.listdir(self.csv_dir)
        csv_data_files = [self.csv_dir + '/' + f for f in csv_data_files]

        logging.info('processing {} csv files from {}'.format(len(csv_data_files), self.csv_dir))

        filtered_data = []
        for i in tqdm(range(len(csv_data_files))):
            data = pd.read_csv(csv_data_files[i])
            filtered_data.append(likelihood_filter(data, self.conf_threshold))
        
        with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'wb') as f:
            joblib.dump(filtered_data, f)

        return filtered_data
    
    def process_features(self, parallel=True):
        with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'rb') as f:
            filtered_data = joblib.load(f)

        n_animals = len(filtered_data)
        logging.info('extracting features from filtered data of {} animals'.format(n_animals))

        if not parallel:
            feats = [extract_bsoid_feats(filtered_data[i]) for i in tqdm(range(n_animals))]
        else:
            from joblib import Parallel, delayed
            feats = Parallel(n_jobs=-1, backend="multiprocessing")(
                    map(delayed(extract_bsoid_feats), filtered_data))

        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'wb') as f:
            joblib.dump(feats, f)

        return feats

    def cluster_feats(self, desired_clusters, n_parts=100, clusters_per_part=100, soft_cluster=True):
        cure = bigCURE(desired_clusters, n_parts, clusters_per_part, soft_cluster)

        with open(self.output_dir + '/' + self.run_id + '_features.sav', 'rb') as f:
            feats = joblib.load(f)

        scaled_feats = normalize_feats(feats)
        clusters = cure.process(scaled_feats)

        with open(self.output_dir + '/' + self.run_id + '_clusters.sav', 'wb') as f:
            joblib.dump(clusters, f)
            