import math
import logging
import itertools
import numpy as np
from BSOID.preprocessing import windowed_feats, smoothen_data

# indices -> features
HEAD, BASE_NECK, CENTER_SPINE, HINDPAW1, HINDPAW2, BASE_TAIL, MID_TAIL, TIP_TAIL = np.arange(8)

FPS = 30

# calculate required features from x, y position data
def extract_feats(filtered_data):
    x, y = filtered_data['x'], filtered_data['y']

    logging.debug('extracting features from {} samples of {} points'.format(*x.shape))

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
    r_thetas = np.zeros_like(thetas)
    r_thetas[:,0] = thetas[:,1] - thetas[:,0]   # between (neck -> center-spine) and (center-spine -> base-tail)
    r_thetas[:,0] = thetas[:,2] - thetas[:,1]   # between (head -> neck) and (neck -> center-spine)
    r_thetas[:,0] = thetas[:,3] - thetas[:,0]   # between (hindpaw1 -> base-tail) and (center-spine -> base-tail)
    r_thetas[:,0] = thetas[:,4] - thetas[:,0]   # between (hindpaw2 -> base-tail) and (center-spine -> base-tail)
    r_thetas[:,0] = thetas[:,5] - thetas[:,0]   # between (mid-tail -> base-tail) and (center-spine -> base-tail)
    r_thetas[:,0] = thetas[:,6] - thetas[:,5]   # between (tip-tail -> mid-tail) and (mid-tail -> base-tail)
    
    # link lengths
    link_lens = np.vstack([np.linalg.norm(np.array(link).T, axis=1) for link in links]).T

    # velocities of points
    # PX_TO_CM = 19.5 * 2.54 / 400
    # velocities = np.hstack([x[:-1] - x[1:], y[:-1] - y[1:]])*PX_TO_CM*30

    return np.hstack((link_lens, r_thetas))
    

def temporal_features(feats, window=16):
    window //= 2
    N = feats.shape[0]

    # extract spectral features over `window` frames
    window_feats = [feats[i - window:i + window] for i in range(window, N - window + 1)]
    spectral_feats = []
    for features in window_feats:
        win_fft = np.fft.rfftn(features, axes=[0])
        spectral_feats.append(win_fft.real ** 2 + win_fft.imag ** 2)
    spectral_feats = np.array(spectral_feats)
    
    # spectral features are of shape (N-M+1, window, d), reshape to (N-M+1, window*d)
    spectral_feats = spectral_feats.reshape(-1, spectral_feats.shape[1]*spectral_feats.shape[2])
    
    return np.hstack((feats[window:-window + 1], spectral_feats))

def extract_bsoid_feats(filtered_data):
    """
    extract the same features as B-SOID:
        - link lengths
        - point displacements
        - angle between link displacements
    """
    x, y = filtered_data['x'], filtered_data['y']
    N, n_dpoints = x.shape

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
            link_dis_cross = np.cross(links[i,j], links[i+1,j])
            curr_angles.append(math.atan2(link_dis_cross, links[i,j].dot(links[i+1,j])))
        angles.append(curr_angles)
    angles = np.array(angles)

    logging.debug('extracted {} link displacement angles from data points'.format(angles.shape[1]))

    # smoothen all features
    for i in range(dis.shape[1]):
        dis[:,i] = smoothen_data(dis[:,i])
    for i in range(link_lens.shape[1]):
        link_lens[:,i] = smoothen_data(link_lens[:,i])
        angles[:,i] = smoothen_data(angles[:,i])

    # window features into bins of a specified number of frames (defaults to 16 frames)
    dis = windowed_feats(dis, mode='mean')
    angles = windowed_feats(angles, mode='sum')
    link_lens = windowed_feats(link_lens, mode='mean')

    feats = np.hstack((dis, angles, link_lens))
    logging.info('final features extracted have shape: {}'.format(feats.shape))

    return feats