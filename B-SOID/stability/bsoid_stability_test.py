import sys
sys.path.insert(0, "/home/laadd/DDP/B-SOID/")

import os
import yaml
import psutil
import shutil
import joblib
import numpy as np
from BSOID.bsoid import BSOID
from BSOID.clustering import *
from BSOID.features import extract_bsoid_feats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

def bsoid_cluster(embeddings, **hdbscan_params):    
    max_classes = -np.infty
    numulab, entropy = [], []

    if "cluster_range" not in hdbscan_params:
        cluster_range = [0.4, 1.2]
    else:
        cluster_range = hdbscan_params["cluster_range"]
        del hdbscan_params["cluster_range"]
    
    if not isinstance(cluster_range, list):
        min_cluster_range = [cluster_range]
    elif len(cluster_range) == 2:
        min_cluster_range = np.linspace(*cluster_range, 25)
    elif len(cluster_range) == 3:
        cluster_range[-1] = int(cluster_range[-1])
        min_cluster_range = np.linspace(*cluster_range)
        
    for min_c in min_cluster_range:
        trained_classifier = hdbscan.HDBSCAN(min_cluster_size=int(round(min_c * 0.01 * embeddings.shape[0])),
                                            **hdbscan_params).fit(embeddings)
        
        labels = trained_classifier.labels_
        numulab.append(labels.max() + 1)
        entropy.append(calculate_entropy_ratio(labels))

        if numulab[-1] > max_classes:
            max_classes = numulab[-1]
            best_clf = trained_classifier

    assignments = best_clf.labels_
    soft_clusters = hdbscan.all_points_membership_vectors(best_clf)
    soft_assignments = np.argmax(soft_clusters, axis=1)

    return assignments, soft_clusters, soft_assignments, best_clf

def get_bsoid_clusters(feats: np.ndarray, hdbscan_params: dict, umap_params: dict):
    embedding = embed_data(feats, scale=True, **umap_params)
    labels, _, soft_labels, _ = bsoid_cluster(embedding, **hdbscan_params)
    return {"labels": labels, "soft_labels": soft_labels}

def bsoid_stabilitytest_train_model(config_file, run_id, base_dir, train_size): 
    bsoid = BSOID(config_file)
    fdata = bsoid.load_filtered_data()
    
    feats, strains = [], list(fdata.keys())
    for strain in strains[:10]:
        feats.extend(Parallel(n_jobs=5)(delayed(extract_bsoid_feats)(data, bsoid.fps, bsoid.stride_window) for data in fdata[strain]))
        del fdata[strain]
    feats = np.vstack(feats)
    
    pca = PCA().fit(StandardScaler().fit_transform(feats))
    ndims = int(np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.7)[0][0] + 1)
    
    feats = feats[np.random.choice(feats.shape[0], train_size, replace=False)]

    bsoid_umap_params = dict(n_neighbors=60, min_dist=0.0, n_components=ndims, metric="euclidean")
    bsoid_hdbscan_params = dict(min_samples=1, prediction_data=True, cluster_range=[0.4, 1.2])

    clustering = get_bsoid_clusters(feats, bsoid_hdbscan_params, bsoid_umap_params)

    model = RandomForestClassifier(n_jobs=1)
    model.fit(StandardScaler().fit_transform(feats), clustering["soft_labels"])
    
    with open(os.path.join(base_dir, f"run-{run_id}.model"), "wb") as f:
        joblib.dump(model, f)

def bsoid_stabilitytest_predictions(models, config_file, test_size, base_dir):
    feats = []
    for _, data in BSOID(config_file).load_features(collect=False).items():
        feats.extend(data)
    feats = np.vstack(feats)

    feats = feats[np.random.choice(feats.shape[0], test_size, replace=False)]
    feats = StandardScaler().fit_transform(feats)

    labels = [clf.predict(feats) for clf in models]
    with open(os.path.join(base_dir, "bsoid_runs.labels"), "wb") as f:
        joblib.dump(labels, f)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser("scripts.py")
    parser.add_argument("--config", type=str)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--base-dir", type=str)
    parser.add_argument("--train-size", type=int)
    args = parser.parse_args()

    config_file = args.config
    run_id = args.run_id
    base_dir = args.base_dir
    train_size = args.train_size

    bsoid_stabilitytest_train_model(config_file, run_id, base_dir, train_size)