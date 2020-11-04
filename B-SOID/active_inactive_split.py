import joblib
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

from BSOID.bsoid import *
from sklearn.preprocessing import StandardScaler
from BSOID.utils import cluster_with_hdbscan

RUN_ID = 'split'
BASE_DIR = '/home/dhruvlaad/data'

def split_data(dis_threshold: float, dis_idx: list):
    bsoid = BSOID.load_config(BASE_DIR, 'dis')

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_features.sav', 'rb') as f:
        feats = joblib.load(f)
    feats = np.vstack(feats)

    # split data according to displacement threshold
    displacements = feats[:,dis_idx].mean(axis=1)

    active_idx = np.where(displacements >= dis_threshold)[0]
    inactive_idx = np.where(displacements < dis_threshold)[0]
    active_feats = feats[active_idx]
    inactive_feats = feats[inactive_idx]

    # create bsoid model for later use
    logging.info(f'divided data into active ({round(active_feats.shape[0]/feats.shape[0], 2)}%) and in-active ({round(inactive_feats.shape[0]/feats.shape[0], 2)}%) based on displacement threshold of {dis_threshold}')
    bsoid = BSOID(RUN_ID, BASE_DIR, fps=30, temporal_dims=None, temporal_window=None, stride_window=3)
    bsoid.save()

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_features.sav', 'wb') as f:
        joblib.dump([active_feats, inactive_feats], f)

def embed_split_data(reduced_dim: int, sample_size: int, parallel=False):
    bsoid = BSOID.load_config(BASE_DIR, RUN_ID)

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_features.sav', 'rb') as f:
        active_feats, inactive_feats = joblib.load(f)

    comb_feats = [active_feats, inactive_feats]

    def embed_subset(feats, sample_size, reduced_dim, umap_params):
        feats_sc = StandardScaler().fit_transform(feats)
    
        # take subset of data
        if sample_size > 0:
            idx = np.random.permutation(np.arange(feats.shape[0]))[0:sample_size]
            feats_train = feats_sc[idx,:]
            feats_usc = feats[idx, :]
        else:
            feats_train = feats_sc
            feats_usc = feats

        logging.info('running UMAP on {} samples from {}D to {}D'.format(*feats_train.shape, reduced_dim))
        mapper = umap.UMAP(n_components=reduced_dim, n_neighbors=100, **umap_params).fit(feats_train)

        return [feats_usc, feats_train, mapper.embedding_]
    
    if parallel:
        from joblib import Parallel, delayed
        umap_results = Parallel(n_jobs=-1)(delayed(embed_subset)(feats, sample_size, reduced_dim, UMAP_PARAMS) for feats in comb_feats)
    else:
        umap_results = [embed_subset(feats, sample_size, reduced_dim, UMAP_PARAMS) for feats in comb_feats]

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_umap.sav', 'wb') as f:
        joblib.dump(umap_results, f)

def cluster_split_data():
    bsoid = BSOID.load_config(BASE_DIR, RUN_ID)

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_umap.sav', 'rb') as f:
        umap_results = joblib.load(f)

    comb_assignments, comb_soft_clusters, comb_soft_assignments = [], [], []
    for results in umap_results:
        _, _, umap_embeddings = results
        
        cluster_range = [float(x) for x in input('Enter cluster range: ').split()]
        assignments, soft_clusters, soft_assignments = cluster_with_hdbscan(umap_embeddings, cluster_range, HDBSCAN_PARAMS)
        comb_assignments.append(assignments)
        comb_soft_clusters.append(soft_clusters)
        comb_soft_assignments.append(soft_assignments)

        logging.info('identified {} clusters from {} samples in {}D'.format(len(np.unique(soft_assignments)), *umap_embeddings.shape))

    n_1, n_2 = comb_assignments[0].size, comb_assignments[1].size
    n_cluster_1, n_cluster_2 = comb_soft_clusters[0].shape[1], comb_soft_clusters[1].shape[1]
    assignments = np.zeros((n_1+n_2,))
    assignments[:n_1] = comb_assignments[0]
    for i in range(n_2):
        if comb_assignments[1][i] >= 0:
            comb_assignments[1][i] += comb_assignments[0].max() + 1
    assignments[n_1:] = comb_assignments[1]

    soft_clusters = -1 * np.ones((n_1 + n_2, n_cluster_1 + n_cluster_2))
    soft_clusters[:n_1,:n_cluster_1] = comb_soft_clusters[0]
    soft_clusters[n_1:, n_cluster_1:] = comb_soft_clusters[1]

    soft_assignments = np.zeros((n_1 + n_2,))
    soft_assignments[:n_1] = comb_soft_assignments[0]
    soft_assignments[n_1:] = comb_soft_assignments[1] + comb_soft_assignments[0].max() + 1

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_clusters.sav', 'wb') as f:
        joblib.dump([assignments, soft_clusters, soft_assignments], f)

    return assignments, soft_clusters, soft_assignments