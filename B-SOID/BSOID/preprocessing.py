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

def likelihood_filter(data: pd.DataFrame, fps, conf_threshold=0.3, end_trim=2, clip_window=30):
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

    filt_x, filt_y = np.zeros_like(x), np.zeros_like(y)
    perc_rect = []

    for i in range(n_dpoints):
        j = 0
        perc_filt = 0

        # find first best point
        while conf[j,i] < conf_threshold and j < N:
            perc_filt += 1
            j += 1
        
        prev_best_point = [x[j,i], y[j,i]]
        filt_x[0:j,i] = prev_best_point[0]*np.ones_like(x[0:j,i])
        filt_y[0:j,i] = prev_best_point[1]*np.ones_like(y[0:j,i])

        for j in range(j, N):
            if conf[j,i] < conf_threshold:
                filt_x[j,i] = prev_best_point[0]
                filt_y[j,i] = prev_best_point[1]
                perc_filt += 1
            else:
                filt_x[j,i], filt_y[j,i] = x[j,i], y[j,i]
                prev_best_point = [x[j,i], y[j,i]]
        
        perc_rect.append(perc_filt / N)
    
    logging.debug(f'filtered {max(perc_rect) * 100}% of data (max)')

    x, y, conf = trim_data(filt_x, filt_y, conf, fps, end_trim, clip_window)

    return {'conf': conf, 'x': x, 'y': y}, max(perc_rect) * 100

def trim_data(x, y, conf, fps, end_trim=2, clip_window=30):
    assert x.shape[1] == y.shape[1]
    assert conf.shape[0] == x.shape[0] == y.shape[0]

    if end_trim > 0:
        end_trim *= (fps * 60)
        conf, x, y = conf[end_trim:-end_trim, :], x[end_trim:-end_trim, :], y[end_trim:-end_trim, :]

    if clip_window > 0:
        try:
            clip_window_ = clip_window // 3
            mid_idx = [i*(conf.shape[0] // 4) for i in range(1, 4)]
            clip_window_ = clip_window_ * 60 * fps // 2
            x_trim, y_trim, conf_trim = [], [], []

            assert mid_idx[0] + clip_window_ < mid_idx[1] - clip_window_
            for idx in mid_idx:
                x_trim.append(x[idx - clip_window_: idx + clip_window_, :])
                y_trim.append(y[idx - clip_window_: idx + clip_window_, :])
                conf_trim.append(conf[idx - clip_window_: idx + clip_window_, :])
            x, y, conf = np.vstack(x_trim), np.vstack(y_trim), np.vstack(conf_trim)
        except AssertionError:
            clip_window = clip_window * 60 * fps // 2
            mid_idx = conf.shape[0] // 2
            conf = conf[mid_idx - clip_window : mid_idx + clip_window, :]
            x = x[mid_idx - clip_window : mid_idx + clip_window, :]
            y = y[mid_idx - clip_window : mid_idx + clip_window, :]

    return (x, y, conf)


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

def windowed_fft(feats, stride_window, temporal_window):
    assert temporal_window > stride_window + 2

    fft_feats = []
    N = feats.shape[0]

    temporal_window = (temporal_window - stride_window) // 2
    for i in range(stride_window + temporal_window, N - temporal_window + 1, stride_window):
        win_feats = feats[i - stride_window - temporal_window:i + temporal_window, :]
        win_fft = np.fft.rfftn(win_feats, axes=[0])
        win_fft = win_fft.real ** 2 + win_fft.imag ** 2
        fft_feats.append(win_fft.reshape(1, -1))

    fft_feats = np.vstack(fft_feats)

    return fft_feats