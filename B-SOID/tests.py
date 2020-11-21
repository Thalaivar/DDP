import umap
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

RUN_ID = 'split'
BASE_DIR = '/home/dhruvlaad/data'
DIS_THRESH = 2.0
SAMPLE_SIZE = -1

import joblib
from joblib import delayed, Parallel
from sklearn.preprocessing import StandardScaler
from active_inactive_split import split_data, calc_dis_threshold
from BSOID.utils import cluster_with_hdbscan
from BSOID.bsoid import BSOID

def embed(feats, feats_sc, n_neighbors, savefile=None):
    if savefile is None:
        savefile = 'umap_test_nbrs'
    if SAMPLE_SIZE > 0 and SAMPLE_SIZE < feats_sc.shape[0]:
        idx = np.random.permutation(np.arange(feats_sc.shape[0]))[0:SAMPLE_SIZE]
        feats_train = feats_sc[idx,:]
        feats_usc = feats[idx, :]
    else:
        feats_train = feats_sc
        feats_usc = feats

    print(f'running UMAP on {feats_train.shape[0]} samples with n_neighbors={n_neighbors}')
    mapper = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.0).fit(feats_train)

    with open(f'/home/dhruvlaad/{savefile}_{n_neighbors}.sav', 'wb') as f:
        joblib.dump([feats_usc, mapper.embedding_], f)

def nbrs_test_all(n_neighbors, parallel=True):
    if not isinstance(n_neighbors, list):
        n_neighbors = [n_neighbors]

    bsoid = BSOID.load_config(base_dir=BASE_DIR, run_id='dis')
    feats, feats_sc = bsoid.load_features()

    print(f'Features have {feats.shape[0]} samples in {feats.shape[1]}D')

    if parallel:
        Parallel(n_jobs=2)(delayed(embed)(feats, feats_sc, nbr, savefile='umap_test_all_nbrs') for nbr in n_neighbors)
    else:  
        [embed(feats, feats_sc, nbr, savefile='umap_test_all_nbrs') for nbr in n_neighbors]

def nbrs_test_active_only(n_neighbors, parallel=True):
    if not isinstance(n_neighbors, list):
        n_neighbors = [n_neighbors]

    with open(f'{BASE_DIR}/output/{RUN_ID}_features.sav', 'rb') as f:
        active_feats, inactive_feats = joblib.load(f)

    print(f'active feats have {active_feats[0].shape[0]} samples and inactive feats have {inactive_feats[0].shape[0]} samples')

    active_feats, active_feats_sc = active_feats
    inactive_feats, inactive_feats_sc = inactive_feats

    displacements = calc_dis_threshold(active_feats)
    assert not np.any(displacements < DIS_THRESH)
    displacements = calc_dis_threshold(inactive_feats)
    assert not np.any(displacements >= DIS_THRESH)

    del inactive_feats, displacements, inactive_feats_sc

    if parallel:
        Parallel(n_jobs=2)(delayed(embed)(active_feats, active_feats_sc, nbr) for nbr in n_neighbors)
    else:  
        [embed(active_feats, active_feats_sc, nbr) for nbr in n_neighbors]


def cluster_test_embeddings(filename, cluster_range):
    # cluster_range = [0.05, 0.5]
    hdbscan_params = {'min_samples': 10, 'prediction_data': True}

    with open(filename, 'rb') as f:
        _, embedding = joblib.load(f)
    
    print(f'clustering {embedding.shape[0]} samples from {filename}')
<<<<<<< HEAD
    assignments, soft_clusters, soft_assignments, best_clf = cluster_with_hdbscan(embedding, cluster_range, hdbscan_params)
=======
    assignments, soft_clusters, soft_assignments, best_clf = cluster_with_hdbscan(embedding, cluster_range, hdbscan_params, detailed=False)
>>>>>>> 69860510b8887d66141e7208d9129618b56a9f20

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
    plt.savefig(filename[:-4]+'.png')

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

<<<<<<< HEAD
    import os
    base_dir = '/Users/dhruvlaad/IIT/DDP/'
    umap_files = [base_dir + f for f in os.listdir(base_dir) if f.endswith('.sav')]
    print(f'running clustering test on {len(umap_files)} files')
    umap_files.sort()

    [cluster_test_embeddings(f, [1, 10]) for f in umap_files]
=======
    filename = 'D:/IIT/DDP/umap_test_nbrs_500.sav'
    cluster_test_embeddings(filename, [0.3, 0.4, 5])
>>>>>>> 69860510b8887d66141e7208d9129618b56a9f20
