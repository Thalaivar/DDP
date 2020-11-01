import os
import joblib
import numpy as np
import pandas as pd

from BSOID.bsoid import BSOID
from BSOID.prediction import *
from BSOID.utils import get_all_bouts
from BSOID.preprocessing import likelihood_filter
from BSOID.features.displacement_feats import *

import logging
logging.basicConfig(level=logging.INFO)

BASE_DIR = 'D:/IIT/DDP/data'
RUN_ID = 'dis'

SAVE_DIR = BASE_DIR + '/analysis'
try:
    os.mkdir(SAVE_DIR)
except FileExistsError:
    for f in os.listdir(SAVE_DIR):
        os.remove(f'{SAVE_DIR}/{f}')

def extract_analysis_data(csv_files):
    bsoid = BSOID.load_config(BASE_DIR, RUN_ID)

    logging.info(f'extracting features and predicting labels for {len(csv_files)} animals')
    # create labels for video
    data = [pd.read_csv(f, low_memory=False) for f in csv_files]
    for i, d in enumerate(data):
        data[i], _ = likelihood_filter(d)
    extract_feats_args = {'stride_window': bsoid.stride_window, 
                'fps': bsoid.fps, 
                'feats_extractor': extract_feats,
                'windower': windowed_feats,
                'temporal_window': bsoid.temporal_window,
                'temporal_dims': bsoid.temporal_dims
            }
    feats = [frameshift_features(d, **extract_feats_args) for d in data]
    with open(f'{bsoid.output_dir}/{bsoid.run_id}_classifiers.sav', 'rb') as f:
        clf = joblib.load(f)
    labels = [frameshift_predict(f, clf, bsoid.stride_window) for f in feats]

    with open(f'{SAVE_DIR}/feats_labels.sav', 'wb') as f:
        joblib.dump([feats, labels, bsoid.fps], f)

def min_transient_bout(min_bout_length=100):
    with open(f'{SAVE_DIR}/feats_labels.sav', 'rb') as f:
        _, labels, fps = joblib.load(f)
    
    # minimum bout length is given in ms, convert to number of frames
    min_bout_length *= (fps / 1000)
    min_bout_length = round(min_bout_length)

    # retain smallest bouts
    class_bouts = get_all_bouts(labels)
    min_transient_bouts = [0 for _ in class_bouts]
    for k, bouts in enumerate(class_bouts):
        for bout in bouts:
            if bout['end'] - bout['start'] <= min_bout_length:
                min_transient_bouts[k] += 1

    min_transient_bouts = [p/len(class_bouts[i]) for i, p in enumerate(min_transient_bouts)]
    return min_transient_bouts