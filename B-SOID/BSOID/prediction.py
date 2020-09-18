import numpy as np
from sklearn.decomposition import PCA
from BSOID.features import extract_feats, temporal_features, window_extracted_feats


def frameshift_features(filtered_data, stride_window, temporal_window, temporal_dims):
    feats = extract_feats(filtered_data)
    # shift by one stride window and extract features
    wfeats = []
    for i in range(stride_window):
        wfeats.append(window_extracted_feats(feats[i:,:], stride_window))
    
    # calculate temporal features
    wfeats = [temporal_features(f, temporal_window) for f in wfeats]
    temporal_feats, feats = [], []
    for f in wfeats:
        feats.append(f[0])
        temporal_feats.append(f[1])

    if temporal_dims is not None:
        # reduce temporal dims
        for i, f in enumerate(temporal_feats):
            pca = PCA(n_components=temporal_dims).fit(f)
            temporal_feats[i] = pca.transform(f)
    
    # feats = [np.hstack((f, sf)) for (f, sf) in (feats, temporal_feats)]
    combined_feats = []
    for i in range(len(feats)):
        combined_feats.append(np.hstack((feats[i], temporal_feats[i])))
    
    return combined_feats

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