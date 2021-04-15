import math
import logging
import numpy as np
from sklearn.decomposition import PCA
from BSOID.preprocessing import (windowed_feats, 
                            smoothen_data,
                            windowed_fft)


def extract_feats(filtered_data, fps, stride_window):
    """
    0-6 : lenghts of 7 body links
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

    logging.debug(f'{angles.shape} displacement angles extracted')
    
    feats = np.hstack((link_lens[1:], dis, angles))

    # for i in range(feats.shape[1]):
    #     feats[:,i] = smoothen_data(feats[:,i], win_len=np.int(np.round(0.05 / (1 / fps)) * 2 - 1))

    logging.debug('extracted {} samples of {}D features'.format(*feats.shape))

    return feats

def window_extracted_feats(feats, stride_window):
    win_feats = []

    for f in feats:
        win_feats_ll_d = windowed_feats(f[:,:7], stride_window, mode='mean')
        win_feats_th = windowed_feats(f[:,7:22], stride_window, mode='sum')
        win_feats.append(np.hstack((win_feats_ll_d, win_feats_th)))
            
    return win_feats