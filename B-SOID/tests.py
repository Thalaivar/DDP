import umap
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

RUN_ID = 'split'
BASE_DIR = '/home/dhruvlaad/data'
DIS_THRESH = 2.0
SAMPLE_SIZE = int(7e5)

import joblib
from joblib import delayed, Parallel
from sklearn.preprocessing import StandardScaler
from active_inactive_split import split_data, calc_dis_threshold
from BSOID.utils import cluster_with_hdbscan

def embed(feats, feats_sc, n_neighbors):
    if SAMPLE_SIZE > 0 and SAMPLE_SIZE < feats_sc.shape[0]:
        idx = np.random.permutation(np.arange(feats_sc.shape[0]))[0:SAMPLE_SIZE]
        feats_train = feats_sc[idx,:]
        feats_usc = feats[idx, :]
    else:
        feats_train = feats_sc
        feats_usc = feats

    print(f'running UMAP on {feats_train.shape[0]} samples with n_neighbors={n_neighbors}')
    mapper = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.0).fit(feats_train)

    with open(f'/home/dhruvlaad/umap_test_nbrs_{n_neighbors}.sav', 'wb') as f:
        joblib.dump([feats_usc, mapper.embedding_], f)

def nbrs_test(n_neighbors):
    if not isinstance(n_neighbors, list):
        n_neighbors = [n_neighbors]

    with open(f'{BASE_DIR}/output/{RUN_ID}_features.sav', 'rb') as f:
        active_feats, inactive_feats = joblib.load(f)

    print(f'active feats have {active_feats.shape[0]} samples and inactive feats have {inactive_feats.shape[0]} samples')
    scaler = StandardScaler().fit(np.vstack((active_feats, inactive_feats)))

    displacements = calc_dis_threshold(active_feats)
    assert not np.any(displacements < DIS_THRESH)
    displacements = calc_dis_threshold(inactive_feats)
    assert not np.any(displacements >= DIS_THRESH)

    del inactive_feats
    del displacements

    feats_sc = scaler.transform(active_feats)
    [embed(active_feats, feats_sc, nbr) for nbr in n_neighbors]

def cluster_test_embeddings(filename, cluster_range):
    # cluster_range = [0.05, 0.5]
    hdbscan_params = {'min_samples': 10, 'prediction_data': True}

    with open(filename, 'rb') as f:
        _, embedding = joblib.load(f)
    
    print(f'clustering {embedding.shape[0]} samples from {filename}')
    assignments, soft_clusters, soft_assignments, best_clf = cluster_with_hdbscan(embedding, cluster_range, hdbscan_params)

    filename = filename[:-4] + '_clusters.sav'
    with open(filename, 'wb') as f:
        joblib.dump([assignments, soft_clusters, soft_assignments, best_clf], f)

    print(f'identified {soft_assignments.max() + 1} clusters saved to {filename}')
    
    labels = np.unique(soft_assignments).astype(np.int64)
    count = [0 for i in labels]
    for label in soft_assignments.astype(np.int64):
        count[label] += 1
    count = [d/soft_assignments.shape[0] for d in count]
    sn.barplot(x=labels, y=count)
    plt.show()