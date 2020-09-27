import logging
import numpy as np
from sklearn.decomposition import PCA
from BSOID.preprocessing import smoothen_data
from sklearn.preprocessing import StandardScaler
from BSOID.features import extract_feats, temporal_features, window_extracted_feats

def frameshift_features(filtered_data, stride_window, temporal_window, temporal_dims, fps, pca=None):
    # extract geometric and temporal features
    feats = extract_feats(filtered_data)
    feats, temporal_feats = temporal_features(feats, temporal_window)
    
    for i in range(feats.shape[1]):
        feats[:,i] = smoothen_data(feats[:,i], win_len=np.int(np.round(0.05 / (1 / fps)) * 2 - 1))

    if temporal_dims is not None and pca is not None:
        logging.debug(f'reducing dimension of temporal features from {temporal_feats.shape[1]}D to {pca.n_features}D')
        # reduce temporal dims
        # pca = PCA(n_components=temporal_dims).fit(temporal_feats)
        temporal_feats = pca.transform(temporal_feats)
    feats = np.hstack((feats, temporal_feats))

    # frameshift and stack features into bins
    fs_feats = []
    for s in range(stride_window):
       fs_feats.append(window_extracted_feats(feats[s:,:], stride_window)) 
    
    # scaling used for classification also
    for i, f in enumerate(fs_feats):
        fs_feats[i] = StandardScaler().fit_transform(f)

    return fs_feats

def frameshift_predict(feats, clf, stride_window):
    labels = []
    for f in feats:
        labels.append(clf.predict(f))

    for n in range(len(labels)):
        labels[n] = labels[n][::-1]
    
    labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
    
    for n, l in enumerate(labels):
        labels_pad[n][0:len(l)] = l
        labels_pad[n] = labels_pad[n][::-1]

        if n > 0:
            labels_pad[n][0:n] = labels_pad[n-1][0:n]
    
    fs_labels = labels_pad.astype(int)
    fs_labels2 = []
    for l in range(stride_window):
        fs_labels2.append(fs_labels[l])
    fs_labels = np.array(fs_labels2).flatten('F')

    return fs_labels