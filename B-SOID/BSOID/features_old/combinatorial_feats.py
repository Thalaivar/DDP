import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
from BSOID.preprocessing import smoothen_data, windowed_feats

def extract_feats(filtered_data, fps, stride_window):
    x_raw, y_raw = filtered_data['x'], filtered_data['y']
    assert x_raw.shape == y_raw.shape
    N, n_dpoints = x_raw.shape

    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1) if stride_window is None else stride_window // 2
    x, y = np.zeros_like(x_raw), np.zeros_like(y_raw)
    for i in range(n_dpoints):
        x[:,i] = smoothen_data(x_raw[:,i], win_len)
        y[:,i] = smoothen_data(y_raw[:,i], win_len)
    
    links = [np.array(x[:,i] - x[:,j], y[:,i] - y[:,j]).T for i, j in combinations(range(n_dpoints), 2)]
    
    # lengths
    link_lens = np.vstack([np.linalg.norm(link, axis=1) for link in links]).T
    link_angles = np.vstack([np.arctan2(link[:,1], link[:,0]) for i, link in enumerate(links)]).T

    return np.hstack((link_angles, link_lens))