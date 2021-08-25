import h5py
import logging
import pandas as pd
import numpy as np
from behaviourPipeline.data import bsoid_format, get_filename_in_dataset, process_h5py_data
from sklearn.preprocessing import StandardScaler

import logging
logger = logging.getLogger(__name__)

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

BODYPARTS = [NOSE_INDEX, LEFT_EAR_INDEX, RIGHT_EAR_INDEX,
            BASE_NECK_INDEX, CENTER_SPINE_INDEX,
            LEFT_REAR_PAW_INDEX, RIGHT_REAR_PAW_INDEX,
            BASE_TAIL_INDEX, MID_TAIL_INDEX, TIP_TAIL_INDEX]

def smoothen_data(data, win_len):
    data = pd.Series(data)
    smoothed_data = data.rolling(win_len, min_periods=1, center=True)
    return np.array(smoothed_data.mean())

def likelihood_filter(data, conf_threshold, bodyparts):
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

    points_filtered_by_idx = np.zeros((n_dpoints,))
    for i in range(n_dpoints):    
        j, perc_filt = 0, 0

        # find first best point
        while j < N and conf[j,i] < conf_threshold:
            perc_filt += 1
            j += 1
        
        filt_x[0:j,i] = np.repeat(x[j, i], j)
        filt_y[0:j,i] = np.repeat(x[j, i], j)
        prev_best_idx = j

        for j in range(j, N):
            if conf[j,i] < conf_threshold:
                filt_x[j,i] = x[prev_best_idx,i]
                filt_y[j,i] = y[prev_best_idx,i]
                perc_filt += 1
            else:
                filt_x[j,i], filt_y[j,i] = x[j,i], y[j,i]
                prev_best_idx = j

        points_filtered_by_idx[i] = perc_filt

    perc_filt = points_filtered_by_idx.max()    
    return {'conf': conf, 'x': filt_x, 'y': filt_y}, perc_filt * 100 / N

def trim_data(fdata: np.ndarray, fps: int):
        # remove first and last 5 mins
        end_trim = 5 * 60 * fps
        fdata = fdata[end_trim:-end_trim]
        
        # consider only first 10 mins
        clip = 10 * 60 * fps
        fdata = fdata[:clip+1]
        
        return fdata

# NOTE: if you change this you should also change the frame extraction function
# def trim_data(x, y, conf, fps, end_trim, clip_window):
#     assert x.shape[1] == y.shape[1]
#     assert conf.shape[0] == x.shape[0] == y.shape[0]

#     # baseline video only 
#     HOUR_LEN = 55 * 60 * fps
#     conf, x, y = conf[:HOUR_LEN, :], x[:HOUR_LEN, :], y[:HOUR_LEN, :]
    
#     if end_trim > 0:
#         end_trim *= (fps * 60)
#         conf, x, y = conf[end_trim:-end_trim, :], x[end_trim:-end_trim, :], y[end_trim:-end_trim, :]

#     if clip_window > 0:            
#             # take first clip_window after trimming
#             clip_window *= (60 * fps)
#             conf = conf[end_trim:end_trim + clip_window, :]
#             x = x[end_trim:end_trim + clip_window, :]
#             y = y[end_trim:end_trim + clip_window, :]

#     return (x, y, conf)

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

def filter_strain_data(df, strain, data_dir, bodyparts, min_video_len, conf_threshold, filter_thresh, n=None):
    if not n: n = df.shape[0]
    df = df.sample(frac=1)

    i, filtered_data = 0, []
    while i < df.shape[0] and len(filtered_data) < n:
        metadata = dict(df.iloc[i])
        i += 1
        try:
            # get location of file in dataset
            filename = get_filename_in_dataset(data_dir, metadata["NetworkFilename"])
            # get data from HDF5 file
            conf, pos = process_h5py_data(h5py.File(filename, "r"))

            if conf.shape[0] < min_video_len: continue
            
            # get raw data in bsoid format in a pandas df
            bsoid_data = bsoid_format(conf, pos)
            
            # filter raw data by confidence threshold
            fdata, perc_filt = likelihood_filter(bsoid_data, conf_threshold, bodyparts)
            strain, mouse_id = metadata['Strain'], metadata['MouseID']
            
            if perc_filt > filter_thresh:
                logger.warning(f'mouse:{strain}/{mouse_id}: % data filtered from raw data is too high ({perc_filt} %)')
                continue

            shape = fdata['x'].shape
            logger.debug(f'preprocessed {shape} data from {strain}/{mouse_id} with {round(perc_filt, 2)}% data filtered')
            filtered_data.append(fdata)

        except: pass

    return strain, filtered_data 