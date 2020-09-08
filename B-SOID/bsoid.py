import os
import random
import joblib
import logging
import pandas as pd
from tqdm import tqdm
from preprocessing import likelihood_filter, extract_bsoid_feats
from data import download_data, conv_bsoid_format

class BSOID:
    def __init__(self, run_id: str, base_dir: str, conf_threshold: float=0.3):
        self.conf_threshold = conf_threshold
        self.run_id = run_id
        self.raw_dir = base_dir + '/raw'
        self.csv_dir = base_dir + '/csv'
        self.output_dir = base_dir + '/output'

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
    
    def process_features(self):
        with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'rb') as f:
            filtered_data = joblib.load(f)
        
        n_animals = len(filtered_data)
        logging.info('extracting features from filtered data of {} animals'.format(n_animals))
        
        for i in tqdm(range(n_animals)):
            feats = extract_bsoid_feats(filtered_data[i])
