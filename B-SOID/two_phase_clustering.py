import os
import umap
import joblib
import numpy as np

from tqdm import tqdm
from BSOID.bsoid import BSOID
from itertools import combinations
from sklearn.metrics import roc_auc_score
from BSOID.utils import cluster_with_hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold

from joblib import Parallel, delayed

strainwise_reduced_dim = 13
strainwise_n_neighbors = 60
strainwise_cluster_rng = [0.5, 1.0, 11]


def collect_strainwise_data(feats):
    for strain, data in feats.items():
        feats[strain] = np.vstack(data)
    return feats

def separate_into_clusters(embedding, labels, feats):
    strains = list(feats.keys())
    feats = collect_strainwise_data(feats)

    clusters, k = {}, 0
    for strain in strains:
        for class_id in np.unique(labels[strain][0]):
            idx = np.where(labels[strain][0] == class_id)[0]
            clusters[f"cluster_{k}"] = {
                "feats": feats[strain][idx],
                "embedding": embedding[strain][idx]
            }
            k += 1
        del feats[strain]
        del embedding[strain]
        del labels[strain]

    return clusters

def pairwise_similarity(cluster1, cluster2):
    X = [cluster1["feats"], cluster2["feats"]]
    y = [np.zeros((X[0].shape[0],)), np.ones((X[1].shape[0],))]

    X, y = np.vstack(X), np.hstack(y)

    model = LinearDiscriminantAnalysis(solver="svd")
    val_score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring="roc_auc")
    
    model.fit(X, y)
    y_pred = model.predict(X)
    sim_score = roc_auc_score(y, y_pred)
    
    return sim_score, val_score

def par_sim_calc(clusters, save_dir, thresh):
    import ray
    import psutil
    
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)

    cluster_labels = list(clusters.keys())

    small_labels = [lab for lab, data in clusters.items() if data["feats"].shape[0] < thresh]
    for lab in small_labels:
        del clusters[lab]
        cluster_labels.remove(lab)
    
    for k, lab in enumerate(cluster_labels):
        clusters[k] = clusters[lab]
        del clusters[lab]
    
    @ray.remote
    def calculate_sim(idx1, idx2, clusters):
        cluster1, cluster2 = clusters[idx1], clusters[idx2]
        sim_score, val_score = pairwise_similarity(cluster1, cluster2)
        return [sim_score, val_score.mean(), idx1, idx2]
    
    cluster_id = ray.put(clusters)
    futures = [calculate_sim.remote(idx1, idx2, cluster_id) for idx1, idx2 in combinations(list(clusters.keys()), 2)]
    
    pbar, sim_data = tqdm(total=len(futures)), []
    while len(futures) > 0:
        num_returns = len(features) if len(features) < num_cpus else num_cpus
        finished, rest = ray.wait(futures, num_returns=num_returns)
        sim_data.extend(ray.get(finished))
        futures = rest
        pbar.update(num_cpus)
        
    sim_data = np.vstack(sim_data)
    sim_mat = np.zeros((len(cluster_labels), len(cluster_labels)))
    val_mat = np.zeros_like(sim_mat)

    for i in range(sim_data.shape[0]):
        sim_score, val_score, idx1, idx2 = sim_data[i]
        idx1, idx2 = int(idx1), int(idx2)
        sim_mat[idx1,idx2], sim_mat[idx2,idx1] = sim_score, sim_score
        val_mat[idx1,idx2], val_mat[idx2,idx1] = val_score, val_score
    
    with open(os.path.join(save_dir, "clusters_sim.sav"), "wb") as f:
        joblib.dump([clusters, sim_mat, val_mat], f)

def generate_sim_matrix(clusters, save_dir, thresh):
    cluster_labels = list(clusters.keys())

    small_labels = [lab for lab, data in clusters.items() if data["feats"].shape[0] < thresh]
    for lab in small_labels:
        del clusters[lab]
        cluster_labels.remove(lab)
    
    for k, lab in enumerate(cluster_labels):
        clusters[k] = clusters[lab]
        del clusters[lab]

    def calculate_sim(idx1, idx2):
        cluster1, cluster2 = clusters[idx1], clusters[idx2]
        sim_score, val_score = pairwise_similarity(cluster1, cluster2)
        return [sim_score, val_score.mean(), idx1, idx2]

    combs = list(combinations(range(len(cluster_labels)), 2))
    sim_data = np.vstack(Parallel(n_jobs=-1)(delayed(calculate_sim)(combs[i][0], combs[i][1]) for i in tqdm(range(len(combs)))))

    sim_mat = np.zeros((len(cluster_labels), len(cluster_labels)))
    val_mat = np.zeros_like(sim_mat)

    for i in range(sim_data.shape[0]):
        sim_score, val_score, idx1, idx2 = sim_data[i]
        idx1, idx2 = int(idx1), int(idx2)
        sim_mat[idx1,idx2], sim_mat[idx2,idx1] = sim_score, sim_score
        val_mat[idx1,idx2], val_mat[idx2,idx1] = val_score, val_score
    
    with open(os.path.join(save_dir, "clusters_sim.sav"), "wb") as f:
        joblib.dump([clusters, sim_mat, val_mat], f)

def strainwise_clustering(config_file, outdir, strain_file=None):
    import logging

    bsoid = BSOID(config_file)
    feats = bsoid.load_features(collect=False)
    feats = collect_strainwise_data(feats)
    
    if strain_file is None:
        strain_list = list(feats.keys())
    else:
        with open(strain_file, "r") as f:
            strain_list = [s[:-1] for s in f.readlines()]

    print(f"Processing {len(strain_list)} strains...")
    embedding, labels = {}, {}
    pbar = tqdm(total=len(strain_list))
    for strain, data in feats.items():
        if strain in strain_list:
            logging.info(f"running for strain: {strain}")
            embedding[strain] = reduce_data(data, n_components=12, n_neighbors=90)
            labels[strain] = cluster_with_hdbscan(embedding[strain], [0.4, 1.2, 25], bsoid.hdbscan_params)
            pbar.update(1)

    with open(os.path.join(outdir, "strainwise_labels.sav"), "wb") as f:
        joblib.dump([embedding, labels], f)

def reduce_data(feats: np.ndarray, n_neighbors: int, n_components: int):
    feats = StandardScaler().fit_transform(feats)

    mapper = umap.UMAP(
            n_components=n_components, 
            min_dist=0.0, 
            n_neighbors=n_neighbors
        ).fit(feats)
    return mapper.embedding_

def find_params(feats):
    import random
    import seaborn as sns
    import matplotlib.pyplot as plt

    strain = random.sample(list(feats.keys()), 1)[0]
    data = feats[strain]
    for strain, data in feats.items():
    # for n_neighbors in range(15, 106, 15):
        # for dim in range(4, 13, 4):
            # print(f"n_neighbors: {n_neighbors} ; dim: {dim}")
        embedding = reduce_data(data, n_neighbors=90, n_components=12)
        labels = cluster_with_hdbscan(embedding, [0.4, 1.0, 25], {"prediction_data": True, "min_samples": 1})[2]

if __name__ == "__main__":
    base_dir = "/home/laadd/data"
    config_file =  os.path.join("./config/config.yaml")

    bsoid = BSOID(config_file)
    bsoid.load_from_dataset()
    bsoid.features_from_points()

    strainwise_clustering(config_file, base_dir)

    with open(os.path.join(base_dir, "strainwise_labels.sav"), "rb") as f:
        embedding, labels = joblib.load(f)

    feats = collect_strainwise_data(bsoid.load_features(collect=False))
    clusters = separate_into_clusters(embedding, labels, feats)

    # generate_sim_matrix(clusters, base_dir, thresh=500)
    par_sim_calc(clusters, base_dir, thresh=1000)