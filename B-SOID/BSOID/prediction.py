import logging
import numpy as np
from sklearn.decomposition import PCA
from BSOID.preprocessing import smoothen_data
from sklearn.preprocessing import StandardScaler

def frameshift_features(filtered_data, stride_window, fps, feats_extractor, windower, temporal_window=None, temporal_dims=None):
    if not isinstance(filtered_data, list):
        filtered_data = [filtered_data]

    feats = [feats_extractor(data, fps) for data in filtered_data]

    assert len(feats) == 1
    feats = feats[0]
    # scaling used for classification also
    feats = StandardScaler().fit_transform(feats)

    # frameshift and stack features into bins
    fs_feats = []
    for s in range(stride_window):
       fs_feats.append(feats[s:,:])

    fs_feats = windower(fs_feats, stride_window, temporal_window, temporal_dims)
    


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