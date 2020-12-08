import math
import logging
import itertools
import numpy as np
from sklearn.decomposition import PCA
from BSOID.preprocessing import (windowed_feats, 
                            smoothen_data,
                            windowed_fft)

def temporal_features(geo_feats, window=16):
    feats, temporal_feats  = [], []
    window //= 2
    for i in range(len(geo_feats)):
        N = geo_feats[i].shape[0]

        # extract spectral features over `window` frames
        window_feats = [geo_feats[i][j - window:j + window] for j in range(window, N - window + 1)]
        spectral_feats = []
        for features in window_feats:
            win_fft = np.fft.rfftn(features, axes=[0])
            spectral_feats.append(win_fft.real ** 2 + win_fft.imag ** 2)
        spectral_feats = np.array(spectral_feats)
        
        # spectral features are of shape (N-M+1, window, d), reshape to (N-M+1, window*d)
        spectral_feats = spectral_feats.reshape(-1, spectral_feats.shape[1]*spectral_feats.shape[2])
        feats.append(geo_feats[i][window:-window + 1])
        temporal_feats.append(spectral_feats)
    
    return feats, temporal_feats

def extract_feats(filtered_data, fps):
    """
    extract the same features as B-SOID:
        0-20 : link lengths
        21-41 : angle between link displacements
        42-48 : displacements (magnitude)
    """
    x, y = filtered_data['x'], filtered_data['y']
    x = np.hstack((x[:,:5], x[:,6:]))
    y = np.hstack((y[:,:5], y[:,6:]))

    N, n_dpoints = x.shape
    # for i in range(x.shape[1]):
    #     x[:,i] = smoothen_data(x[:,i], win_len=np.int(np.round(0.05 / (1 / fps)) * 2 - 1))
    #     y[:,i] = smoothen_data(y[:,i], win_len=np.int(np.round(0.05 / (1 / fps)) * 2 - 1))

    # displacements of all points
    dis = []
    for i in range(N - 1):
        dis_vec = np.array([x[i+1,:] - x[i,:], y[i+1,:] - y[i,:]])
        dis.append(np.linalg.norm(dis_vec, axis=0))
    dis = np.array(dis)

    logging.debug('extracted {} displacements of {} data points'.format(dis.shape[0], dis.shape[1]))

    # links of all possible combinations
    links = []
    for k in range(N):
        curr_links = []
        for i, j in itertools.combinations(range(n_dpoints), 2):
            curr_links.append([x[k,i] - x[k,j], y[k,i] - y[k,j]])
        links.append(curr_links)
    links = np.array(links)

    logging.debug('extracted {} links from data points'.format(links.shape[1]))

    # lengths of links
    link_lens = []
    for i in range(N):
        link_lens.append(np.linalg.norm(links[i,:,:], axis=1))
    link_lens = np.array(link_lens)

    logging.debug('extracted {} link lengths from data points'.format(link_lens.shape[1]))

    # angles between link position for two timesteps
    angles = []
    for i in range(N-1):
        curr_angles = []
        for j in range(links.shape[1]):
            link_dis_cross = np.cross(links[i+1,j], links[i,j])
            # curr_angles.append(math.atan2(link_dis_cross, links[i,j].dot(links[i+1,j])))
            th =  np.dot(np.dot(np.sign(link_dis_cross), 180) / np.pi,
                        math.atan2(np.linalg.norm(link_dis_cross),
                        np.dot(links[i+1,j], links[i,j])))
            curr_angles.append(th)
        angles.append(curr_angles)
    angles = np.array(angles)

    logging.debug('extracted {} link displacement angles from data points'.format(angles.shape[1]))

    # smoothen all features
    for i in range(dis.shape[1]):
        dis[:,i] = smoothen_data(dis[:,i], win_len=np.int(np.round(0.05 / (1 / fps)) * 2 - 1))
    for i in range(link_lens.shape[1]):
        link_lens[:,i] = smoothen_data(link_lens[:,i], win_len=np.int(np.round(0.05 / (1 / fps)) * 2 - 1))
        angles[:,i] = smoothen_data(angles[:,i], win_len=np.int(np.round(0.05 / (1 / fps)) * 2 - 1))

    feats = np.hstack((link_lens[1:], angles, dis))
    logging.debug('final features extracted have shape: {}'.format(feats.shape))

    return feats

def window_extracted_feats(feats, stride_window, temporal_window=None, temporal_dims=None):
    win_feats = []

    for f in feats:
        if temporal_window is not None:
            # indices 0-6 are link lengths, during windowing they should be averaged
            clip_len = (temporal_window - stride_window) // 2
            
            win_feats_ll = windowed_feats(f[clip_len:-clip_len,0:21], stride_window, mode='mean')
            win_feats_th = windowed_feats(f[clip_len:-clip_len,21:], stride_window, mode='sum')

            win_fft = windowed_fft(f, stride_window, temporal_window)
        
            if temporal_dims is not None:
                win_fft = PCA(n_components=temporal_dims).fit_transform(win_fft)
            
            win_feats.append(np.hstack((win_feats_ll, win_feats_th, win_fft)))

        else:
            win_feats_ll = windowed_feats(f[:,0:21], stride_window, mode='mean')
            win_feats_th = windowed_feats(f[:,21:], stride_window, mode='sum')
            win_feats.append(np.hstack((win_feats_ll, win_feats_th)))
            
    return win_feats