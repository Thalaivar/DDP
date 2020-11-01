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
    pass

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
                'windower': window_extracted_feats,
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
    bouts_by_animal = [get_all_bouts(x) for x in labels]
    class_bouts = [[] for _ in range(len(bouts_by_animal[0]))]
    for i in range(len(bouts_by_animal)):
        for j, bouts in enumerate(bouts_by_animal[i]):
            class_bouts[j].extend(bouts_by_animal[i][j])
    min_transient_bouts = [0 for _ in class_bouts]
    for k, bouts in enumerate(class_bouts):
        for bout in bouts:
            if bout['end'] - bout['start'] <= min_bout_length:
                min_transient_bouts[k] += 1

    min_transient_bouts = [p/len(class_bouts[i]) for i, p in enumerate(min_transient_bouts)]
    return min_transient_bouts

def fit_markov_models():
    with open(f'{SAVE_DIR}/feats_labels.sav', 'rb') as f:
        _, labels, fps = joblib.load(f)

    n = len(labels)
    n_clusters = max([len(np.unique(x)) for x in labels])
    
    # compute zero-th order probabilities
    markov_0th_order = np.zeros((n_clusters,))
    total_samples = 0
    for i in range(n):
        for idx in labels[i]:
            markov_0th_order[idx] += 1
            total_samples += 1
    
    markov_0th_order /= total_samples

    # compute first-order probabilities
    markov_1st_order = np.zeros((n_clusters,n_clusters))
    for i in range(n):
        for j in range(len(labels[i]) - 1):
            idx1, idx2 = labels[i][j], labels[i][j+1]
            markov_1st_order[idx1, idx2] += 1
    
    for i in range(n_clusters):
        
if __name__ == "__main__":
    # csv_files = [BASE_DIR + '/test/' + f for f in os.listdir(BASE_DIR+'/test') if f.endswith('.csv')]
    # extract_analysis_data(csv_files)
    print(min_transient_bout())