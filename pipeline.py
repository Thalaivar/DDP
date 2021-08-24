import os
import joblib
import yaml
import psutil
import random
import pandas as pd
from preprocessing import filter_strain_data
from joblib import Parallel, delayed

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
        
        for d in [self.base_dir, self.output_dir, self.csv_dir, self.raw_dir]:
            try: os.mkdir(d)
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

        filtered_data = [(strain, data) for strain, data in filtered_data if len(data) > 0]
        if n_strains > 0: filtered_data = random.sample(filtered_data)
        filtered_data = {strain: data for strain, data in filtered_data}

        logger.info(f"extracted data from {len(filtered_data)} strains with a total of {sum(len(data) for _, data in filtered_data.items())} animals")
        self.save_to_cache(filtered_data, "filtered_strainwise_data.sav")

        return filtered_data

    def save_to_cache(self, data, f):
        with open(os.path.join(self.base_dir, f), "wb") as fname:
            joblib.dump(data, fname)
    
    def load(self, f):
        with open(os.path.join(self.base_dir, f), "rb") as fname:
            data = joblib.load(fname)
        return data