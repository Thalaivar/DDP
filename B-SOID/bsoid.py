import os
import joblib
import logging
import pandas as pd
from tqdm import tqdm
from preprocessing import likelihood_filter

class BSOID:
    def __init__(self, run_id: str, base_dir: str, conf_threshold: float=0.3):
        self.conf_threshold = conf_threshold
        self.run_id = run_id
        self.csv_dir = base_dir + '/csv'
        self.output_dir = base_dir + '/output'

        try:
            os.mkdir(self.output_dir)    
        except FileExistsError:
            pass

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
        