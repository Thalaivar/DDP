import math
import logging
import numpy as np
from sklearn.decomposition import PCA
from BSOID.preprocessing import (windowed_feats, 
                            smoothen_data,
                            windowed_fft)

def extract_feats(filtered_data, fps):
    x, y = filtered_data['x'], filtered_data['y']

    logging.debug('extracting features from {} samples of {} points'.format(*x.shape))

    # indices -> features
    HEAD, BASE_NECK, CENTER_SPINE, HINDPAW1, HINDPAW2, BASE_TAIL, MID_TAIL, TIP_TAIL = np.arange(8)
    
    # link connections [start, end]
    link_connections = ([BASE_TAIL, CENTER_SPINE],
                        [CENTER_SPINE, BASE_NECK], 
                        [BASE_NECK, HEAD],
                        [BASE_TAIL, HINDPAW1], [BASE_TAIL, HINDPAW2],
                        [BASE_TAIL, MID_TAIL],
                        [BASE_TAIL, TIP_TAIL])

    # links between body points and the angles they make with x-axis
    links = []
    for conn in link_connections:
        links.append([x[:, conn[0]] - x[:, conn[1]], y[:, conn[0]] - y[:, conn[1]]])
    thetas = []
    for link in links:
        thetas.append(np.arctan2(link[1], link[0]))
    thetas = np.vstack(thetas).T

    logging.debug('{} links and {} angles extracted'.format(len(links), thetas.shape))

    # relative angles between links
    r_thetas = np.zeros((thetas.shape[0], thetas.shape[1]-1))
    r_thetas[:,0] = thetas[:,1] - thetas[:,0]   # between (neck -> center-spine) and (center-spine -> base-tail)
    r_thetas[:,1] = thetas[:,2] - thetas[:,1]   # between (head -> neck) and (neck -> center-spine)
    r_thetas[:,2] = thetas[:,3] - thetas[:,0]   # between (hindpaw1 -> base-tail) and (center-spine -> base-tail)
    r_thetas[:,3] = thetas[:,4] - thetas[:,0]   # between (hindpaw2 -> base-tail) and (center-spine -> base-tail)
    r_thetas[:,4] = thetas[:,5] - thetas[:,0]   # between (mid-tail -> base-tail) and (center-spine -> base-tail)
    r_thetas[:,5] = thetas[:,6] - thetas[:,5]   # between (tip-tail -> mid-tail) and (mid-tail -> base-tail)
    
    # link lengths
    link_lens = np.vstack([np.linalg.norm(np.array(link).T, axis=1) for link in links]).T

    feats = np.hstack((link_lens, r_thetas))
    
    # smoothen data
    for i in range(feats.shape[1]):
        feats[:,i] = smoothen_data(feats[:,i], win_len=np.int(np.round(0.05 / (1 / fps)) * 2 - 1))
    
    return feats

def window_extracted_feats(feats, stride_window, temporal_window=None, temporal_dims=None):
    win_feats = []

    for f in feats:
        if temporal_window is not None:
            # indices 0-6 are link lengths, during windowing they should be averaged
            clip_len = (temporal_window - stride_window) // 2
            
            win_feats_ll = windowed_feats(f[clip_len:-clip_len+1,:7], stride_window, mode='mean')
            win_feats_rth = windowed_feats(f[clip_len:-clip_len+1,7:], stride_window, mode='sum')

            win_fft = windowed_fft(f, stride_window, temporal_window)
        
            if temporal_dims is not None:
                win_fft = PCA(n_components=temporal_dims).fit_transform(win_fft)
            
            win_feats.append(np.hstack((win_feats_ll, win_feats_rth, win_fft)))

        else:
            win_feats_ll = windowed_feats(f[:,:7], stride_window, mode='mean')
            win_feats_rth = windowed_feats(f[:,7:], stride_window, mode='sum')
            win_feats.append(np.hstack((win_feats_ll, win_feats_rth)))
            
    return win_feats