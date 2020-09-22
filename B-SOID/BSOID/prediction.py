import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from BSOID.features import extract_feats, temporal_features, window_extracted_feats

def frameshift_features(filtered_data, stride_window, temporal_window, temporal_dims, fps):
    # extract geometric and temporal features
    feats = extract_feats(filtered_data)
    feats, temporal_feats = temporal_features(feats, temporal_window)
    
    if temporal_dims is not None:
        # reduce temporal dims
        for i, f in enumerate(temporal_feats):
            pca = PCA(n_components=temporal_dims).fit(f)
            temporal_feats[i] = pca.transform(f)
    feats = np.hstack((feats, temporal_feats))

    # frameshift and stack features into bins
    fs_feats = []
    for s in range(stride_window):
       fs_feats.append(window_extracted_feats(feats[s:,:], stride_window)) 
    
    # scaling used for classification also
    # for i, f in enumerate(fs_feats):
    #     fs_feats[i] = StandardScaler().fit_transform(f)

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