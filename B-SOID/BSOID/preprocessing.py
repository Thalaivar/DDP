import pandas as pd
import numpy as np
import logging
import math
import itertools
from sklearn.preprocessing import StandardScaler

def smoothen_data(data, win_len=7):
    data = pd.Series(data)
    smoothed_data = data.rolling(win_len, min_periods=1, center=True)
    return np.array(smoothed_data.mean())

def likelihood_filter(data: pd.DataFrame, conf_threshold: float=0.3, forward_fill=True):
    N = data.shape[0]
    n_dpoints = data.shape[1] // 3
    # retrieve confidence, x and y data from csv data
    conf, x, y = [], [], []
    for col in data.columns:
        if col.endswith('_lh'):
            conf.append(data[col])
        elif col.endswith('_x'):
            x.append(data[col])
        elif col.endswith('_y'):
            y.append(data[col])
    conf, x, y = np.array(conf).T, np.array(x).T, np.array(y).T

    logging.debug('extracted {} samples of {} features'.format(N, n_dpoints))

    # forward-fill any points below confidence threshold
    if forward_fill:
        perc_filt = []
        for i in range(n_dpoints):
            n_filtered = 0
            if conf[0,i] < conf_threshold:
                # find first good confidence point
                k = 0
                while conf[k,i] < conf_threshold:
                    n_filtered += 1
                    k += 1
                # replace all points with first good conf point
                conf[0:k,i] = conf[k,i]*np.ones_like(conf[0:k,i])
                x[0:k,i] = x[k,i]*np.ones_like(x[0:k,i])
                y[0:k,i] = y[k,i]*np.ones_like(y[0:k,i])
                
                prev_lh_idx = k
            else:
                prev_lh_idx = 0                
                k = 0

            for j in range(k, N):
                if conf[j,i] < conf_threshold:
                    # if current point is low confidence, replace with last confident point
                    x[j,i], y[j,i] = x[prev_lh_idx,i], y[prev_lh_idx,i]
                    n_filtered += 1
                else:
                    prev_lh_idx = j
        
            perc_filt.append(n_filtered)
        perc_filt = [(p/N)*100 for p in perc_filt]
    
        logging.debug('%% filtered from all features: {}'.format(perc_filt))
    
    return {'conf': conf, 'x': x, 'y': y}

def windowed_feats(feats, window_len: int=3, mode: str='mean'):
    """
    collect features over a window of `window_len` frames
    """
    win_feats = []
    N = feats.shape[0]

    logging.debug('collecting {} frames into bins of {} frames'.format(N, window_len))

    for i in range(window_len, N, window_len):
        if mode is 'mean':
            win_feats.append(feats[i-window_len:i,:].mean(axis=0))
        elif mode is 'sum':
            win_feats.append(feats[i-window_len:i,:].sum(axis=0))

    return np.array(win_feats)
    
def normalize_feats(feats):
    logging.info('normalizing features from {} animals with sklearn StandardScaler()'.format(len(feats)))
    scaled_feats = []

    scaler = StandardScaler()
    for i in range(len(feats)):
        scaler.fit(feats[i])
        scaled_feats.append(scaler.transform(feats[i]))

    return np.vstack(scaled_feats)