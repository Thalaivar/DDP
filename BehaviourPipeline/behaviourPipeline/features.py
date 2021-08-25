import numpy as np
from itertools import combinations
from behaviourPipeline.preprocessing import windowed_feats, smoothen_data

"""
B-SOID original
"""
def extract_bsoid_feats(filtered_data, fps, stride_window):
    x, y = filtered_data['x'], filtered_data['y']
    assert x.shape == y.shape
    N, n_dpoints = x.shape

    # extract geometric features
    disp = np.linalg.norm(np.array([x[1:,:] - x[0:N-1,:], y[1:,:] - y[0:N-1,:]]), axis=0)
    links = [np.array([x[:,i] - x[:,j], y[:,i] - y[:,j]]).T for i, j in combinations(range(n_dpoints), 2)]
    ll = np.vstack([np.linalg.norm(link, axis=1) for link in links]).T
    dis_angles = np.vstack([np.arctan2(np.cross(link[0:N-1], link[1:]), np.sum(link[0:N-1] * link[1:], axis=1)) for link in links]).T

    # smoothen features over 100ms window    
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    for i in range(ll.shape[1]):
        ll[:,i] = smoothen_data(ll[:,i], win_len)
        dis_angles[:,i] = smoothen_data(dis_angles[:,i], win_len)
    for i in range(disp.shape[1]):
        disp[:,i] = smoothen_data(disp[:,i], win_len)

    ll = windowed_feats(ll, stride_window, mode="mean")
    dis_angles = windowed_feats(dis_angles, stride_window, mode="sum")
    disp = windowed_feats(disp, stride_window, mode="sum")
    
    return np.hstack((ll, dis_angles, disp))

def extract_comb_feats(filtered_data: dict, fps: int):
    """
    Extract geometric features from all possible combinations of keypoints. The features 
    extracted are as follows:

        For every pair of keypoints:
            - ll          : length of the link formed by the pair
            - link_angles : angle made by the link with x - axis
            - dis_angles  : displacement of angle made by link
    
    Further the displacements of the keypoints themselves are also added:
        - disp : norm of displacement of keypoints

    Inputs:
        - filtered_data : preprocessed keypoint data with keys ['x', 'y'] for x and y coordinates of each keypoint
        - fps           : frames-per-second  
    Outputs:
        - feats : ( N x 3 * Dc2 + D ) array of geometric features where D is no. of. keypoints
    """
    x, y = filtered_data['x'], filtered_data['y']
    assert x.shape == y.shape
    N, n_dpoints = x.shape

    # extract geometric features
    disp = np.linalg.norm(np.array([x[1:,:] - x[0:N-1,:], y[1:,:] - y[0:N-1,:]]), axis=0)
    links = [np.array([x[:,i] - x[:,j], y[:,i] - y[:,j]]).T for i, j in combinations(range(n_dpoints), 2)]
    link_angles = np.vstack([np.arctan2(link[:,1], link[:,0]) for link in links]).T
    ll = np.vstack([np.linalg.norm(link, axis=1) for link in links]).T
    dis_angles = np.vstack([np.arctan2(np.cross(link[0:N-1], link[1:]), np.sum(link[0:N-1] * link[1:], axis=1)) for link in links]).T

    # smoothen features over 100ms window    
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    for i in range(ll.shape[1]):
        ll[:,i] = smoothen_data(ll[:,i], win_len)
        dis_angles[:,i] = smoothen_data(dis_angles[:,i], win_len)
        link_angles[:,i] = smoothen_data(link_angles[:,i], win_len)
    for i in range(disp.shape[1]):
        disp[:,i] = smoothen_data(disp[:,i], win_len)

    return np.hstack((ll[1:], link_angles[1:], dis_angles, disp))

# NOTE: If you change the feature extraction function to include/exclude (or even change the ordering of) any features you must make appropriate changes here
def aggregate_features(feats: np.ndarray, stride_window: int):
    """
    Aggregate features into bins to improve behaviour discovery
    Inputs:
        - feats         : ( N x 3 * Dc2 + D ) array of geometric features where D is no. of. keypoints
        - stride_window : no. of frames over which the aggregation is done
    Outputs:
        - win_feats     : ( (N / stride_window) x 3 * Dc2 + D ) aggregated features 
    """

    # identify number of keypoints
    D = int((1 + np.sqrt(1 + 24 * feats.shape[1])) / 6)
    n_pairs = int(D * (D - 1) / 2)

    # geometric features
    ll, link_angles, dis_angles, disp = feats[:,:n_pairs], feats[:,n_pairs:2*n_pairs], feats[:,2*n_pairs:3*n_pairs], feats[:,3*n_pairs:]
    
    # average link lengths and angles
    ll = windowed_feats(ll, stride_window, mode="mean")
    link_angles = windowed_feats(link_angles, stride_window, mode="mean")

    # sum displacement features
    dis_angles = windowed_feats(dis_angles, stride_window, mode="sum")
    disp = windowed_feats(disp, stride_window, mode="sum")

    return np.hstack((ll, link_angles, dis_angles, disp))