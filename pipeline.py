import os
import joblib
import yaml
import psutil
import random
import pandas as pd
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed
from preprocessing import filter_strain_data, trim_data
from features import extract_comb_feats, aggregate_features

import logging
logger = logging.getLogger(__name__)

class BehaviourPipeline:
    def __init__(self, config: str):
        with open(config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.pipelinename     = config["pipelinename"]
        base_dir        = os.path.join(config["base_dir"], self.pipelinename)
        self.base_dir   = base_dir

        self.fps            = config["fps"]
        self.stride_window  = round(config["stride_window"] * self.fps / 1000)
        self.conf_threshold = config["conf_threshold"]
        self.bodyparts      = config["bodyparts"]
        self.filter_thresh  = config["filter_thresh"]
        self.min_video_len  = config["min_video_len"]
        
    
        try: os.mkdir(self.base_dir)
        except FileExistsError: pass
        
    def ingest_data(self, data_dir: str, records: pd.DataFrame, n: int, n_strains: int=-1, n_jobs: int=-1):
        min_video_len = self.min_video_len * self.fps * 60
        
        n_jobs = min(n_jobs, psutil.cpu_count(logical=False))
        filtered_data = Parallel(n_jobs)(
            delayed(filter_strain_data)(
                df,
                strain,
                data_dir, 
                self.bodyparts, 
                min_video_len, 
                self.conf_threshold, 
                self.filter_thresh, 
                n
            )
            for strain, df in list(records.groupby("Strain"))
        )

        filtered_data = [(strain, animal_data) for strain, animal_data in filtered_data if len(animal_data) > 0]
        if n_strains > 0: filtered_data = random.sample(filtered_data, n_strains)
        filtered_data = {strain: animal_data for strain, animal_data in filtered_data}

        # trim filtered data
        for strain, animal_data in filtered_data.items():
            for i, fdata in enumerate(animal_data):
                for key, data in fdata.items():
                    fdata[key] = trim_data(data, self.fps)
                animal_data[i] = fdata
            filtered_data[strain] = animal_data

        logger.info(f"extracted data from {len(filtered_data)} strains with a total of {sum(len(data) for _, data in filtered_data.items())} animals")
        self.save_to_cache(filtered_data, "strains.sav")

        return filtered_data

    def compute_features(self, n_jobs: int=-1):
        n_jobs = min(n_jobs, psutil.cpu_count(logical=False))

        filtered_data = self.load("strains.sav")
        logger.info(f'extracting features from {len(filtered_data)} strains')

        pbar = tqdm(total=len(filtered_data))
        feats = {}

        for strain, fdata in filtered_data.items():
            feats[strain] = Parallel(n_jobs)(delayed(extract_comb_feats)(data, self.fps) for data in fdata)
            feats[strain] = [aggregate_features(f, self.stride_window) for f in feats[strain]]
            pbar.update(1)
        
        logger.info(f'extracted {len(feats)} datasets of {feats[list(feats.keys())[0]][0].shape[1]}D features')
        logger.info(f'collected features into bins of {1000 * self.stride_window // self.fps} ms')

        self.save_to_cache(feats, "features.sav")
        return feats
        
    def save_to_cache(self, data, f):
        with open(os.path.join(self.base_dir, f), "wb") as fname:
            joblib.dump(data, fname)
    
    def load(self, f):
        with open(os.path.join(self.base_dir, f), "rb") as fname:
            data = joblib.load(fname)
        return data