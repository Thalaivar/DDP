import math
import logging
import numpy as np
from sklearn.decomposition import PCA
from BSOID.preprocessing import (windowed_feats, 
                            smoothen_data,
                            windowed_fft)


def extract_feats(filtered_data, fps):
    """
    0-6 : lenghts of 7 body links
    7-12 : relative angles of links
    13-20 : magnitude of displacements for all 8 points
    21-27 : displacement angles for links 
    """
    x, y = filtered_data['x'], filtered_data['y']

    N, n_dpoints = x.shape
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

    # displacement of points
    dis = np.array([x[1:,:] - x[0:N-1,:], y[1:,:] - y[0:N-1,:]])
    dis = np.linalg.norm(dis, axis=0)

    # links
    links = []
    for conn in link_connections:
        links.append(np.array([x[:, conn[0]] - x[:, conn[1]], y[:, conn[0]] - y[:, conn[1]]]).T)    

    thetas = []
    for link in links:
        thetas.append(np.arctan2(link[:,0], link[:,1]))
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
    r_thetas += np.pi
    r_thetas %= 2*np.pi

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
    
    feats = np.hstack((link_lens[1:], r_thetas[1:], dis, angles))
    
    # smoothen data
    for i in range(feats.shape[1]):
        feats[:,i] = smoothen_data(feats[:,i], win_len=np.int(np.round(0.05 / (1 / fps)) * 2 - 1))

    logging.debug('extracted {} samples of {}D features'.format(*feats.shape))

    return feats

def window_extracted_feats(feats, stride_window, temporal_window=None, temporal_dims=None):
    win_feats = []

    for f in feats:
        if temporal_window is not None:
            # indices 0-6 are link lengths, during windowing they should be averaged
            clip_len = (temporal_window - stride_window) // 2
            
            win_feats_ll_d = windowed_feats(f[clip_len:-clip_len+1,:21], stride_window, mode='mean')
            win_feats_th = windowed_feats(f[clip_len:-clip_len+1,21:28], stride_window, mode='sum')

            win_fft = windowed_fft(f, stride_window, temporal_window)
        
            if temporal_dims is not None:
                win_fft = PCA(n_components=temporal_dims).fit_transform(win_fft)
            
            win_feats.append(np.hstack((win_feats_ll_d, win_feats_th, win_fft)))

        else:
            win_feats_ll_d = windowed_feats(f[:,:21], stride_window, mode='mean')
            win_feats_th = windowed_feats(f[:,21:28], stride_window, mode='sum')
            win_feats.append(np.hstack((win_feats_ll_d, win_feats_th)))
            
    return win_feats