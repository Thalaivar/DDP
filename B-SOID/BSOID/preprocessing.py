import pandas as pd
import numpy as np
import logging
import math
import itertools
from sklearn.preprocessing import StandardScaler

# Mapping for data
NOSE_INDEX = 0
LEFT_EAR_INDEX = 1
RIGHT_EAR_INDEX = 2
BASE_NECK_INDEX = 3
LEFT_FRONT_PAW_INDEX = 4
RIGHT_FRONT_PAW_INDEX = 5
CENTER_SPINE_INDEX = 6
LEFT_REAR_PAW_INDEX = 7
RIGHT_REAR_PAW_INDEX = 8
BASE_TAIL_INDEX = 9
MID_TAIL_INDEX = 10
TIP_TAIL_INDEX = 11

bodyparts = [NOSE_INDEX, LEFT_EAR_INDEX, RIGHT_EAR_INDEX,
            BASE_NECK_INDEX, CENTER_SPINE_INDEX,
            LEFT_REAR_PAW_INDEX, RIGHT_REAR_PAW_INDEX,
            BASE_TAIL_INDEX, MID_TAIL_INDEX, TIP_TAIL_INDEX]

def smoothen_data(data, win_len=7):
    data = pd.Series(data)
    smoothed_data = data.rolling(win_len, min_periods=1, center=True)
    return np.array(smoothed_data.mean())

def likelihood_filter(data: pd.DataFrame, conf_threshold: float=0.3, forward_fill=True):
    N = data.shape[0]

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
    conf, x, y = conf[:,bodyparts], x[:,bodyparts], y[:,bodyparts]
    
    # take average of nose and ears
    conf = np.hstack((conf[:,:3].mean(axis=1).reshape(-1,1), conf[:,3:]))
    x = np.hstack((x[:,:3].mean(axis=1).reshape(-1,1), x[:,3:]))
    y = np.hstack((y[:,:3].mean(axis=1).reshape(-1,1), y[:,3:]))

    n_dpoints = conf.shape[1]
    
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

                    if k >= conf.shape[0]:
                        return None, 100

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
    
        logging.debug('% filtered from all features (max): {}'.format(max(perc_filt)))
    
    return {'conf': conf, 'x': x, 'y': y}, max(perc_filt)

def windowed_feats(feats, window_len: int=3, mode: str='mean'):
    """
    collect features over a window of `window_len` frames
    """
    win_feats = []
    N = feats.shape[0]

    logging.debug('collecting {} frames into bins of {} frames'.format(N, window_len))

    for i in range(window_len, N, window_len):
        if mode == 'mean':
            win_feats.append(feats[i-window_len:i,:].mean(axis=0))
        elif mode == 'sum':
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

def windowed_fft(feats, stride_window, temporal_window):
    assert temporal_window > stride_window + 2

    fft_feats = []
    N = feats.shape[0]

    temporal_window = (temporal_window - stride_window) // 2
    for i in range(stride_window + temporal_window, N - temporal_window, stride_window):
        win_feats = feats[i - stride_window - temporal_window:i + temporal_window, :]
        win_fft = np.fft.rfftn(win_feats, axes=[0])
        win_fft = win_fft.real ** 2 + win_fft.imag ** 2
        fft_feats.append(win_fft.reshape(1, -1))

    fft_feats = np.vstack(fft_feats)

    return fft_feats