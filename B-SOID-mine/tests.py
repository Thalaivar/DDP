import umap
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

RUN_ID = 'dis'
BASE_DIR = '/home/dhruvlaad/data'
DIS_THRESH = 2.0
SAMPLE_SIZE = int(3e5)

import joblib
from joblib import delayed, Parallel
from sklearn.preprocessing import StandardScaler
from active_inactive_split import split_data, calc_dis_threshold
from BSOID.utils import cluster_with_hdbscan
from BSOID.bsoid import BSOID

def embed(feats, feats_sc, n_neighbors, savefile=False):
    if SAMPLE_SIZE > 0 and SAMPLE_SIZE < feats_sc.shape[0]:
        idx = np.random.permutation(np.arange(feats_sc.shape[0]))[0:SAMPLE_SIZE]
        feats_train = feats_sc[idx,:]
        feats_usc = feats[idx, :]
    else:
        feats_train = feats_sc
        feats_usc = feats

    print(f'running UMAP on {feats_train.shape[0]} samples with n_neighbors={n_neighbors}')
    mapper = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.0).fit(feats_train)

    if savefile is not False:
        with open(f'/home/dhruvlaad/{savefile}_{n_neighbors}.sav', 'wb') as f:
            joblib.dump([feats_usc, mapper.embedding_], f)

    return mapper.embedding_

def nbrs_test_all(n_neighbors, run_id='dis', parallel=True):
    if not isinstance(n_neighbors, list):
        n_neighbors = [n_neighbors]

    bsoid = BSOID.load_config(base_dir=BASE_DIR, run_id=run_id)
    feats, feats_sc = bsoid.load_features()

    print(f'Features have {feats.shape[0]} samples in {feats.shape[1]}D')

    if parallel:
        embeddings = Parallel(n_jobs=-1)(delayed(embed)(feats, feats_sc, nbr, savefile=False) for nbr in n_neighbors)
    else:  
        embeddings = [embed(feats, feats_sc, nbr, savefile=False) for nbr in n_neighbors]
    
    save_file = f'/home/dhruvlaad/{run_id}_feats_embeddings_2d.pkl'    
    with open(save_file, 'wb') as f:
        joblib.dump([embeddings, n_neighbors], f)

    return embeddings

def nbrs_test_active_only(n_neighbors, parallel=True):
    if not isinstance(n_neighbors, list):
        n_neighbors = [n_neighbors]

    with open(f'{BASE_DIR}/output/split_features.sav', 'rb') as f:
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
        embeddings = Parallel(n_jobs=2)(delayed(embed)(active_feats, active_feats_sc, nbr) for nbr in n_neighbors)
    else:  
        embeddings = [embed(active_feats, active_feats_sc, nbr) for nbr in n_neighbors]

    save_file = f'/home/dhruvlaad/split_feats_embeddings_2d.pkl'    
    with open(save_file, 'wb') as f:
        joblib.dump([embeddings, n_neighbors], f)

def cluster_test_embeddings(embedding, cluster_range):
    # cluster_range = [0.05, 0.5]
    hdbscan_params = {'min_samples': 10, 'prediction_data': True}
    
    assignments, soft_clusters, soft_assignments, best_clf = cluster_with_hdbscan(embedding, cluster_range, hdbscan_params)

    print(f'identified {soft_assignments.max() + 1} clusters')
    return assignments, soft_assignments

def cluster_nbrs_test(embeddings_data_file):
    with open(embeddings_data_file, 'rb') as f:
        embeddings, nbrs = joblib.load(f)

    print(f'Saved files contain {len(embeddings)} saved datasets with {embeddings[0].shape} samples for n_neighbors={nbrs}')

    labels = []
    for e in embeddings:
        labs, _ = cluster_test_embeddings(e, cluster_range=[0.2, 1.0, 8])
        labels.append(labs)
    
    with open(f'{embeddings_data_file[:-4]}_clusters.pkl', 'wb') as f:
        joblib.dump(labels, f)
    
    for i in range(len(labels)):
        embeddings[i] = embeddings[i][labels[i] >= 0]
        labels[i] = labels[i][labels[i] >= 0]

    nrows, ncols = [int(x) for x in input(f'Enter (nrows, ncols): ').split()]
    fig, ax = plt.subplots(nrows, ncols)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            if k < len(embeddings):
                if nrows > 1:
                    ax[i,j].scatter(embeddings[k][:,0], embeddings[k][:,1], s=1, label=f'nbrs={nbrs[k]}', c=labels[k])
                    ax[i,j].legend(loc='upper right')
                else:
                    ax[j].scatter(embeddings[k][:,0], embeddings[k][:,1], s=1, label=f'nbrs={nbrs[k]}', c=labels[k])
                    ax[j].legend(loc='upper right')
                k += 1
    
    fig.show()

def plot_2d_embeddings(embeddings_data_file):
    with open(embeddings_data_file, 'rb') as f:
        embeddings, nbrs = joblib.load(f)
    
    nrows, ncols = 1, 3
    fig, ax = plt.subplots(nrows, ncols)
    k = 0
    if nrows > 1:
        for i in range(nrows):
            for j in range(ncols):
                if k < len(embeddings):
                    ax[i,j].scatter(embeddings[k][:,0], embeddings[k][:,1], s=0.1, alpha=0.01, label=f'nbrs={nbrs[k]}')
                    ax[i,j].legend(loc='upper right')
                    k += 1
    else:
        for i in range(ncols):
            if k < len(embeddings):
                ax[i].scatter(embeddings[k][:,0], embeddings[k][:,1], s=0.1, alpha=0.01, label=f'nbrs={nbrs[k]}')
                ax[i].legend(loc='upper right')
                k += 1
    
    fig.show()

def plot_2d_embedding_w_labels(embedding_file, label_file):
    with open(embedding_file, 'rb') as f:
        embeddings, nbrs = joblib.load(f)
    
    with open(label_file, 'rb') as f:
        labels = joblib.load(f)
    
    assert len(labels) == len(embeddings)
    
    for i in range(len(labels)):
        embeddings[i] = embeddings[i][labels[i] >= 0]
        labels[i] = labels[i][labels[i] >= 0]

    nrows, ncols = [int(x) for x in input(f'Saved files contain {len(embeddings)} saved datasets with {embeddings[0].shape} samples, enter (nrows, ncols): ').split()]

    fig, ax = plt.subplots(nrows, ncols)
    k = 0
    if nrows > 1:
        for i in range(nrows):
            for j in range(ncols):
                if k < len(embeddings):
                    ax[i,j].scatter(embeddings[k][:,0], embeddings[k][:,1], s=0.1, alpha=0.01, label=f'nbrs={nbrs[k]}', c=labels[k])
                    ax[i,j].legend(loc='upper right')
                    k += 1
    else:
        for i in range(ncols):
            if k < len(embeddings):
                ax[i].scatter(embeddings[k][:,0], embeddings[k][:,1], s=0.1, alpha=0.01, label=f'nbrs={nbrs[k]}', c=labels[k])
                ax[i].legend(loc='upper right')
                k += 1
    
    fig.show()

    props = []
    for label in labels:
        prop = [0 for _ in range(label.max() + 1)]
        for idx in label:
            if idx >= 0:
                prop[idx] += 1
        prop = np.array(prop)
        prop = prop/prop.sum()
        props.append(prop)

    fig, ax = plt.subplots(nrows, ncols)
    k = 0
    if nrows > 1:
        for i in range(nrows):
            for j in range(ncols):
                if k < len(embeddings):
                    sn.barplot(ax=ax[i,j], x=np.arange(labels[k].max() + 1), y=props[k])
                    ax[i,j].set_title(f'n_neighbors = {nbrs[k]}')
                    k += 1
    else:
        for i in range(ncols):
            if k < len(embeddings):
                sn.barplot(ax=ax[i], x=np.arange(labels[k].max() + 1), y=props[k])
                ax[i].set_title(f'n_neighbors = {nbrs[k]}')
                k += 1

    fig.show()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    import os
    base_dir = 'D:/IIT/DDP/data/nbrs_test/'
    # umap_files = [base_dir + f for f in os.listdir(base_dir) if f.endswith('.sav')]
    # umap_files.sort()

    umap_files = ['umap_test_all_nbrs_150.sav', 'umap_test_all_nbrs_200.sav', 'umap_test_all_nbrs_250.sav', 'umap_test_all_nbrs_300.sav']
    umap_files = [base_dir + f for f in umap_files]
    
    print(f'Running clustering test on {umap_files}')

    [cluster_test_embeddings(f, [0.1, 1.0, 10]) for f in umap_files]

    # umap_file = base_dir + 'umap_test_nbrs_300.sav'
    # cluster_test_embeddings(umap_file, cluster_range=[5, 10, 5])
