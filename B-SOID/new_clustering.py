import umap
import joblib
import logging
import numpy as np

from tqdm import tqdm
from sklearn import clone
from BSOID.bsoid import BSOID
from itertools import combinations
from sklearn.metrics import roc_auc_score
from BSOID.utils import cluster_with_hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate

STRAINWISE_UMAP_PARAMS = {
    "n_neighbors": 90,
    "n_components": 12
}
STRAINWISE_CLUSTER_RNG = [0.4, 1.2, 25]

HDBSCAN_PARAMS = {"prediction_data": True, "min_samples": 1}
THRESH = 1000

DISC_MODEL = LinearDiscriminantAnalysis(solver="svd")
CV = StratifiedKFold(n_splits=5, shuffle=True)

GROUPWISE_UMAP_PARAMS = {
    "n_neighbors": 60,
    "n_components": 3
}
GROUPWISE_CLUSTER_RNG = [0.5, 4, 25]

CLF = RandomForestClassifier(class_weight="balanced", n_jobs=-1)

def reduce_data(feats: np.ndarray):
    feats = StandardScaler().fit_transform(feats)
    mapper = umap.UMAP(min_dist=0.0, **STRAINWISE_UMAP_PARAMS).fit(feats)
    return mapper.embedding_

def collect_strainwise_feats(feats: dict):
    for strain, animal_data in feats.items():
        feats[strain] = np.vstack(animal_data)
    return feats

def cluster_strainwise(config_file, save_dir):
    bsoid = BSOID(config_file)
    feats = collect_strainwise_feats(bsoid.load_features(collect=False))

    print(f"Processing {len(feats)} strains...")
    
    embedding, labels, pbar = {}, {}, tqdm(total=len(feats))
    for strain, data in feats.items():
        logging.info(f"running for strain: {strain}")
        embedding[strain] = reduce_data(data)
        results = cluster_with_hdbscan(embedding[strain], STRAINWISE_CLUSTER_RNG, HDBSCAN_PARAMS)
        labels[strain] = [results[2], results[3]]
        pbar.update(1)
    
    return embedding, labels

def collect_strainwise_clusters(feats: dict, labels: dict, embedding: dict):
    k, clusters, strain2cluster = 0, {}, {strain: [] for strain in feats.keys()}
    for strain in feats.keys():
        for class_id in np.unique(labels[strain][0]):
            idx = np.where(labels[strain][0] == class_id)[0]
            if idx.size >= THRESH:
                clusters[k] = {
                    "feats": feats[strain][idx,:],
                    "embed": embedding[strain][idx,:]
                }
                strain2cluster[strain].append(k)
                k += 1
    
    return clusters, strain2cluster

def cluster_similarity(cluster1, cluster2):
    X = [cluster1["feats"], cluster2["feats"]]
    y = [np.zeros((X[0].shape[0],)), np.ones((X[1].shape[0], ))]
    X, y = np.vstack(X), np.hstack(y)

    model = clone(DISC_MODEL)
    val_score = cross_val_score(model, X, y, cv=CV, scoring="roc_auc")

    model.fit(X, y)
    y_pred = model.predict(X)
    score = roc_auc_score(y, y_pred)
    
    return score, val_score.mean()

def pairwise_similarity(feats, embedding, labels):
    import ray
    import psutil

    feats = collect_strainwise_feats(feats)
    clusters, strain2clusters = collect_strainwise_clusters(feats, labels, embedding)
    del feats, embedding, labels

    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)

    @ray.remote
    def par_pwise(idx1, idx2, clusters):
        sim, val = cluster_similarity(clusters[idx1], clusters[idx2])
        return [sim, val, idx1, idx2]
    
    clusters_id = ray.put(clusters)
    futures = [par_pwise.remote(idx1, idx2, clusters_id) 
                    for idx1, idx2 in 
                    combinations(list(clusters.keys()), 2)]
    
    pbar, sim = tqdm(total=len(futures)), []
    while len(futures) > 0:
        n = len(futures) if len(futures) < num_cpus else num_cpus
        fin, rest = ray.wait(futures, num_returns=n)
        sim.extend(ray.get(fin))
        futures = rest
        pbar.update(n)
    
    sim = np.vstack(sim)
    return sim, strain2clusters

def group_clusters_together(feats, labels, embedding, sim):
    feats = collect_strainwise_feats(feats)
    clusters = collect_strainwise_clusters(feats, labels, embedding)
    del feats, embedding, labels

    mapper = umap.UMAP(min_dist=0.0, **GROUPWISE_UMAP_PARAMS).fit(sim)
    group_labels = cluster_with_hdbscan(mapper.embedding_, GROUPWISE_CLUSTER_RNG, HDBSCAN_PARAMS)[2]

    grouped_clusters = {}
    for cluster_id, group_id in enumerate(group_labels):
        if group_id in grouped_clusters:
            grouped_clusters[group_id] = np.vstack([
                    grouped_clusters[group_id]["feats"],
                    clusters[cluster_id]["feats"]
                ])
        else:
            grouped_clusters[group_id] = clusters[cluster_id]["feats"]

    return grouped_clusters

def learn_classifier(grouped_clusters):
    X, y = [], []

    for class_id, data in grouped_clusters.items():
        X.append(data)
        y.append(class_id * np.ones(data.shape[0]))
    
    X, y = np.vstack(X), np.hstack(y)

    model = clone(CLF)
    val_score = cross_validate(model, X, y, cv=CV, scoring=("f1_weighted"))["test_f1_weighted"].mean()

    model.fit(X, y)
    return model, val_score

def get_counts(clusters):
    return {name: data.shape[0] for name, data in clusters.items()}

if __name__ == "__main__":
    import os
    import logging
    logging.basicConfig(level=logging.INFO)
    
    save_dir = "/home/laadd/data"
    config_file = "./config/config.yaml"

    # bsoid = BSOID(config_file)
    # bsoid.load_from_dataset(n=10)
    # feats = bsoid.load_features(collect=False)

    # embedding, labels = cluster_strainwise(config_file, save_dir)

    # with open(os.path.join(save_dir, "strainwise_ckpt1.sav"), "wb") as f:
    #     joblib.dump([feats, embedding, labels], f)

    with open(os.path.join(save_dir, "strainwise_ckpt1.sav"), "rb") as f:
        feats, embedding, labels = joblib.load(f)
        
    sim, strain2clusters = pairwise_similarity(feats, embedding, labels)

    with open(os.path.join(save_dir, "strainwise.sav"), "wb") as f:
        joblib.dump([feats, embedding, labels, sim, strain2clusters], f)
    
    # os.remove(os.path.join(save_dir, "strainwise_ckpt1.sav"))