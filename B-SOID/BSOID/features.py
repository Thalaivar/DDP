import math
import logging
import psutil
import numpy as np
from itertools import combinations
from BSOID.preprocessing import windowed_feats, smoothen_data
from behavelet import wavelet_transform

def extract_dis_feats(filtered_data, fps, stride_window):
    """
    0-6 : lenghts ofW 7 body links
    7-14 : magnitude of displacements for all 8 points
    14-21 : displacement angles for links 
    """
    x_raw, y_raw = filtered_data['x'], filtered_data['y']

    assert x_raw.shape == y_raw.shape
    N, n_dpoints = x_raw.shape

    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1) if stride_window is None else stride_window // 2
    logging.debug('feature extraction from {} samples of {} points with {} ms smoothing window'.format(*x_raw.shape, round(win_len * 1000 / 30)))
    
    x, y = np.zeros_like(x_raw), np.zeros_like(y_raw)
    for i in range(x.shape[1]):
        x[:,i] = smoothen_data(x_raw[:,i], win_len)
        y[:,i] = smoothen_data(y_raw[:,i], win_len)

    # indices -> features
    HEAD, BASE_NECK, CENTER_SPINE, HINDPAW1, HINDPAW2, BASE_TAIL, MID_TAIL, TIP_TAIL = np.arange(8)

    # link connections [start, end]
    link_connections = ([BASE_TAIL, CENTER_SPINE],
                        [CENTER_SPINE, BASE_NECK],
                        [BASE_NECK, HEAD],
                        [BASE_TAIL, HINDPAW1], [BASE_TAIL, HINDPAW2],
                        [BASE_TAIL, MID_TAIL],
                        [MID_TAIL, TIP_TAIL])

    # displacement of points
    dis = np.array([x[1:,:] - x[0:N-1,:], y[1:,:] - y[0:N-1,:]])
    dis = np.linalg.norm(dis, axis=0)

    # links
    links = []
    for conn in link_connections:
        links.append(np.array([x[:, conn[0]] - x[:, conn[1]], y[:, conn[0]] - y[:, conn[1]]]).T)    

    # link lengths
    link_lens = np.vstack([np.linalg.norm(link, axis=1) for link in links]).T

    # angles between link position for consecutive timesteps
    angles = []
    for link in links:
        curr_angles = []
        for k in range(N-1):
            link_dis_cross = np.cross(link[k], link[k+1])
            curr_angles.append(math.atan2(link_dis_cross, link[k].dot(link[k+1])))
        angles.append(np.array(curr_angles))
    angles = np.vstack(angles).T
    
    feats = np.hstack((link_lens[1:], dis, angles))
    
    win_feats = []
    win_feats.append(windowed_feats(feats[:,:7], stride_window, mode='mean'))
    win_feats.append(windowed_feats(feats[:,7:22], stride_window, mode='sum'))
    win_feats = np.hstack(win_feats)

    return win_feats

def extract_comb_feats(filtered_data: dict, fps: int, stride_window: int):
    x, y = filtered_data['x'], filtered_data['y']
    assert x.shape == y.shape
    N, n_dpoints = x.shape

    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    
    disp = np.linalg.norm(np.array([x[1:,:] - x[0:N-1,:], y[1:,:] - y[0:N-1,:]]), axis=0)
    links = [np.array([x[:,i] - x[:,j], y[:,i] - y[:,j]]).T for i, j in combinations(range(n_dpoints), 2)]
    ll = np.vstack([np.linalg.norm(link, axis=1) for link in links]).T
    dis_angles = np.vstack([np.arctan2(np.cross(link[0:N-1], link[1:]), np.sum(link[0:N-1] * link[1:], axis=1)) for link in links]).T
    
    for i in range(ll.shape[1]):
        ll[:,i] = smoothen_data(ll[:,i], win_len)
        dis_angles[:,i] = smoothen_data(dis_angles[:,i], win_len)
    for i in range(disp.shape[1]):
        disp[:,i] = smoothen_data(disp[:,i], win_len)

    ll = windowed_feats(ll, stride_window, mode="mean")
    dis_angles = windowed_feats(dis_angles, stride_window, mode="sum")
    disp = windowed_feats(disp, stride_window, mode="sum")

    return np.hstack((ll, dis_angles, disp))

def extract_temporal_feats(filtered_data: dict, fps: int, stride_window: int):
    x, y = filtered_data['x'], filtered_data['y']
    N, n_dpoints = x.shape
    
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    
    links = [np.array([x[:,i] - x[:,j], y[:,i] - y[:,j]]).T for i, j in combinations(range(n_dpoints), 2)]
    ll = np.vstack([np.linalg.norm(link, axis=1) for link in links]).T
    ll_disp = ll[1:] - ll[:N-1]
    disp = np.linalg.norm(np.array([x[1:,:] - x[0:N-1,:], y[1:,:] - y[0:N-1,:]]), axis=0)
    ll_disp_th = np.vstack([np.arctan2(np.cross(link[0:N-1], link[1:]), np.sum(link[0:N-1] * link[1:], axis=1)) for link in links]).T
    link_angles = np.vstack([np.arctan2(link[:,1], link[:,0]) for link in links]).T
    
    for i in range(ll.shape[1]):
        ll[:,i] = smoothen_data(ll[:,i], win_len)
        ll_disp_th[:,i] = smoothen_data(ll_disp_th[:,i], win_len)
        link_angles[:,i] = smoothen_data(link_angles[:,i], win_len)
    for i in range(disp.shape[1]):
        disp[:,i] = smoothen_data(disp[:,i], win_len)
    
    feats = np.hstack((ll, link_angles))
    feats = wavelet_transform(feats, n_jobs=psutil.cpu_count(logical=False), n_freqs=25, fmin=0.15, fmax=15, fsample=30)[2]

    feats = feats[stride_window:-1:stride_window]
    return feats    