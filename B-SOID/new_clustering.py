import os
import umap
import joblib
import numpy as np

from tqdm import tqdm
from numba import njit
from sklearn import clone
from BSOID.bsoid import BSOID
from BSOID.utils import max_entropy
from itertools import combinations
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from BSOID.utils import cluster_with_hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate

import logging
logger = logging.getLogger(__name__)

HDBSCAN_PARAMS = {"prediction_data": True, "min_samples": 1}

CV = StratifiedKFold(n_splits=5, shuffle=True)

GROUPWISE_UMAP_PARAMS = {
    "n_neighbors": 60,
    "n_components": 3
}
GROUPWISE_CLUSTER_RNG = [1, 5, 25]

def reduce_data(feats: np.ndarray, **umap_params):
    feats = StandardScaler().fit_transform(feats)
    mapper = umap.UMAP(min_dist=0.0, **umap_params).fit(feats)
    return mapper.embedding_

def collect_strainwise_feats(feats: dict):
    for strain, animal_data in feats.items():
        feats[strain] = np.vstack(animal_data)
    return feats

def cluster_strainwise(config_file, save_dir, logfile):
    import ray
    import psutil

    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)
    logger.info(f"running on: {num_cpus} CPUs")

    @ray.remote
    def cluster_strain_data(strain, feats):
        logger = open(logfile, "a")
        data = feats[strain]

        logger.write(f"running for strain: {strain} with samples: {data.shape}\n")

        strainwise_umap_params = {"n_neighbors": 90, "n_components": 25}
        strainwise_cluster_rng = [0.4, 1.2, 25]
        hdbscan_params = {"prediction_data": True, "min_samples": 1, "core_dist_n_jobs": 1}
        
        embedding = reduce_data(data, **strainwise_umap_params)
        assignments, _, soft_assignments, _ = cluster_with_hdbscan(embedding, strainwise_cluster_rng, hdbscan_params)
        
        prop = [p / soft_assignments.size for p in np.unique(soft_assignments, return_counts=True)[1]]
        entropy_ratio = -sum(p * np.log2(p) for p in prop) / max_entropy(assignments.max() + 1)

        logger.write(f"collected {embedding.shape[0]} samples for {strain} with {assignments.max() + 1} classes and entropy ratio: {entropy_ratio}\n")
        return (strain, embedding, (assignments, soft_assignments))
    
    bsoid = BSOID(config_file)
    feats = collect_strainwise_feats(bsoid.load_features(collect=False))
    feats_id = ray.put(feats)

    print(f"Processing {len(feats)} strains...")
    futures = [cluster_strain_data.remote(strain, feats_id) for strain in feats.keys()]
    
    pbar, results = tqdm(total=len(futures)), []
    while len(futures) > 0:
        n = len(futures) if len(futures) < num_cpus else num_cpus
        fin, rest = ray.wait(futures, num_returns=n)
        results.extend(ray.get(fin))
        futures = rest
        pbar.update(n)

    embedding, labels = {}, {}
    for res in results:
        strain, embed, lab = res
        embedding[strain] = embed
        labels[strain] = lab

    return embedding, labels

def collect_strainwise_labels(feats, embedding, labels):
    import copy
    new_labels = copy.deepcopy(labels)
    for strain, (assignments, soft_assignments) in labels.items():
        # feats[strain] = feats[strain][assignments >= 0]
        # embedding[strain] = embedding[strain][assignments >= 0]
        # labels[strain] = soft_assignments[assignments >= 0]

        new_labels[strain] = soft_assignments
        logger.info(f"Strain: {strain} ; Features: {feats[strain].shape} ; Embedding: {embedding[strain].shape} ; Labels: {new_labels[strain].shape}")
    
    return feats, embedding, new_labels

def collect_strainwise_clusters(feats: dict, labels: dict, embedding: dict, thresh: float):
    feats = collect_strainwise_feats(feats)
    feats, embedding, labels = collect_strainwise_labels(feats, embedding, labels)

    k, clusters = 0, {}
    for strain in feats.keys():
        labels[strain] = labels[strain].astype(int)

        # threshold by entropy
        n = labels[strain].max() + 1
        class_ids, counts = np.unique(labels[strain], return_counts=True)
        prop = [x/labels[strain].size for x in counts]
        entropy_ratio = -sum(p * np.log2(p) for p in prop) / max_entropy(n)

        if entropy_ratio >= thresh:
            logger.info(f"pooling {len(class_ids)} clusters from {strain} with entropy ratio {entropy_ratio}")
            for class_id in class_ids:
                idx = np.where(labels[strain] == class_id)[0]
                clusters[f"{strain}:{class_id}:{k}"] = {
                    "feats": feats[strain][idx,:],
                    "embed": embedding[strain][idx,:]
                }
                k += 1
    
    return clusters

def get_strain2cluster_map(clusters):
    strain2cluster = {}
    for cluster_id in clusters.keys():
        strain, _, k = cluster_id.split(':')
        if strain in strain2cluster:
            strain2cluster[strain].append(int(k))
        else:
            strain2cluster[strain] = [int(k)]
    return strain2cluster

@njit
def cdist2sim(D):
    m, n = D.shape
    result = 0.0
    for i in range(m):
        for j in range(i, n):
            result += D[i,j]
    return result / (m * n)

def pairwise_similarity(feats, embedding, labels, thresh):
    import ray
    import psutil

    clusters = collect_strainwise_clusters(feats, labels, embedding, thresh)
    print(f"Total clusters: {len(clusters)}")
    del feats, embedding, labels

    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)
    logger.info(f"running on: {num_cpus} CPUs")
    
    @ray.remote
    def par_pwise(idx1, idx2, X1, X2):
        D = cdist(X1, X2, metric="euclidean")
        sim = cdist2sim(D)
        
        idx1, idx2 = int(idx1.split(':')[-1]), int(idx2.split(':')[-1])
        return [sim, idx1, idx2]
    
    # clusters_id = ray.put(clusters)
    # logger.info("initializing tasks")
    # futures = [par_pwise.remote(idx1, idx2, clusters[idx1]["feats"], clusters[idx2]["feats"]) 
    #                 for idx1, idx2 in 
    #                 combinations(list(clusters.keys()), 2)]
    
    pwise_combs = list(combinations(list(clusters.keys()), 2))
    k, pbar, sim = 0, tqdm(total=len(pwise_combs)), []
    futures = []
    while k < len(pwise_combs):
        for i in range(k, min(k+num_cpus, len(pwise_combs))):
            idx1, idx2 = pwise_combs[i]
            X1, X2 = clusters[idx1]["feats"], clusters[idx2]["feats"]
            futures.append(par_pwise.remote(idx1, idx2, X1, X2))
        fin, futures = ray.wait(futures, num_returns=min(num_cpus, len(futures)), timeout=3000)
        sim.extend(ray.get(fin))
        pbar.update(len(fin))
    
    sim = np.vstack(sim)
    return sim

def impute_same_strain_values(sim, clusters):
    strain2cluster = get_strain2cluster_map(clusters)

    idxmap = {}
    for _, idxs in strain2cluster.items():
        for idx in idxs:
            idxmap[idx] = [i for i in idxs if i != idx]
    
    retain_idx, same_strain_idx = [], []
    for i in range(sim.shape[0]):
        if int(sim[i,2]) in idxmap[int(sim[i,1])]:
            same_strain_idx.append(i)
        else:
            retain_idx.append(i)
    
    diff_strain_sim = sim[retain_idx]

    from sklearn.neighbors import KNeighborsRegressor
    filler = KNeighborsRegressor(n_neighbors=20, n_jobs=-1, metric="canberra", weights="distance")
    filler.fit(diff_strain_sim[:,1:].astype(int), diff_strain_sim[:,0])
    
    same_strain_sim = filler.predict(sim[same_strain_idx, 1:].astype(int))
    same_strain_sim = np.hstack((same_strain_sim.reshape(-1,1), sim[same_strain_idx,1:]))

    sim = np.vstack((diff_strain_sim, same_strain_sim))
    return sim

def similarity_matrix(sim):
    n_clusters = int(sim[:,1:].max()) + 1
    sim[:,0] = np.abs(sim[:,0] - 0.5) + 0.5
    sim[:,0] = (sim[:,0] - 0.5) / 0.5
    
    mat = np.zeros((n_clusters, n_clusters))

    for i in range(sim.shape[0]):
        idx1, idx2 = sim[i,1:].astype("int")
        mat[idx1,idx2] = mat[idx2,idx1] = sim[i,0]
    
    return mat

def group_clusters(clusters, sim, strain2cluster):
    mapper = umap.UMAP(min_dist=0.0, **GROUPWISE_UMAP_PARAMS).fit(similarity_matrix(sim, strain2cluster))
    _, _, glabels, _ = cluster_with_hdbscan(mapper.embedding_, GROUPWISE_CLUSTER_RNG, HDBSCAN_PARAMS)

    groups = {}
    for cluster_idx, group_idx in enumerate(glabels.astype("int")):
        if group_idx in groups:
            groups[group_idx] = np.vstack((groups[group_idx], clusters[cluster_idx]["feats"]))
        else:
            groups[group_idx] = clusters[cluster_idx]["feats"]

    return groups       

def train_classifier(groups, **params):
    X, y = [], []
    for lab, feats in groups.items():
        X, y = X + [feats], y + [lab * np.ones((feats.shape[0],))]
    X, y = np.vstack(X), np.hstack(y).astype(int)
    
    idx = np.random.permutation(np.arange(y.size))
    X, y = X[idx], y[idx]

    _, counts = np.unique(y, return_counts=True)
    import matplotlib.pyplot as plt
    plt.bar(range(len(counts)), counts)
    plt.show()

    if input("Undersampling [yes/no]: ") == "yes":
        lab, counts = np.unique(y, return_counts=True)
        counts = {l: n if n < int(counts.mean()) else int(counts.mean()) for l, n in zip(lab, counts)}
        from imblearn.under_sampling import RandomUnderSampler
        X, y = RandomUnderSampler(sampling_strategy=counts).fit_resample(X, y)
        _, counts = np.unique(y, return_counts=True)
        plt.bar(range(len(counts)), counts)
        plt.show()
    
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(loss_function="MultiClass", eval_metric="Accuracy", iterations=3000, task_type="GPU", verbose=True, learning_rate=0.1)
    model.set_params(**params)
    
    # model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # preds = model.predict(X_test)
    # from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score
    # print("f1_score: ", f1_score(y_test, preds, average="weighted"))
    # print("roc_auc_score: ", roc_auc_score(y_test, model.predict_proba(X_test), average="weighted", multi_class="ovo"))
    # print("accuracy: ", accuracy_score(y_test, preds))
    # print("balanced_acc: ", balanced_accuracy_score(y_test, preds))
    
    model.fit(X, y)
    return model

if __name__ == "__main__":
    with open("../../data/undersampling/cluster_collect_embed.sav", "rb") as f:
        groups = joblib.load(f)
    
    model = train_classifier(groups)
    with open("../../data/dis/output/dis_classifiers.sav", "wb") as f:
        joblib.dump(model, f)