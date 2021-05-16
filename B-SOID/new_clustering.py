import os
import re
import umap
import joblib
import hdbscan
import numpy as np

from tqdm import tqdm
from numba import njit
from sklearn import clone
from BSOID.bsoid import BSOID
from BSOID.similarity import *
from BSOID.utils import max_entropy, calculate_entropy_ratio
from itertools import combinations
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from BSOID.utils import cluster_with_hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate

from BSOID.features import *

import logging
logger = logging.getLogger(__name__)

HDBSCAN_PARAMS = {"prediction_data": True, "min_samples": 1}

CV = StratifiedKFold(n_splits=5, shuffle=True)

GROUPWISE_UMAP_PARAMS = {
    "n_neighbors": 60,
    "n_components": 3
}
GROUPWISE_CLUSTER_RNG = [1, 5, 25]

def reduce_data(feats: np.ndarray):
    # feats = StandardScaler().fit_transform(feats)
    mapper = umap.UMAP(min_dist=0.0, n_neighbors=60, n_components=20, metric="symmetric_kl").fit(feats)
    return mapper.embedding_

def get_clusters(feats: np.ndarray, verbose=False):
    embedding = reduce_data(feats)
    labels, _, soft_labels, clusterer = cluster_with_hdbscan(embedding, [0.4, 1.2], {"prediction_data": True, "min_samples": 1}, verbose=verbose)
    exemplars = [feats[idxs] for idxs in clusterer.exemplars_indices_]
    
    logger.info(f"embedded {feats.shape} to {embedding.shape[1]}D with {labels.max() + 1} clusters and entropyiip ratio={round(calculate_entropy_ratio(soft_labels),3)}")
    return {"labels": labels, "soft_labels": soft_labels, "exemplars": exemplars}

def sample_points_from_clustering(labels, feats, n):
    classes, counts = np.unique(labels, return_counts=True)

    # probabilty of selecting a group is inversely proportional to its density
    props = np.array([1 / (counts[i] / labels.size) for i in range(counts.size)])
    props = props / props.sum()
    
    # get labels for sampled points
    rep_point_labels = np.random.choice(classes, size=n, replace=True, p=props)
    rep_classes, rep_counts = np.unique(rep_point_labels, return_counts=True)
    
    # get sampled points
    data = []
    for i in range(rep_classes.size):
        idxs = np.where(labels == rep_classes[i])[0]
        logger.info(f"sampling {rep_counts[i]} from {idxs.size} points in class {rep_classes[i]}")
        if rep_counts[i] > idxs.size:
            rep_counts[i] = idxs.size
        
        data.append(feats[np.random.choice(idxs, rep_counts[i], replace=False)])

    return np.vstack(data)

def cluster_for_strain(feats: list, n: int, parallel=False, verbose=False):
    if parallel:
        import psutil
        from joblib import Parallel, delayed
        clustering = Parallel(n_jobs=psutil.cpu_count(logical=False))(delayed(get_clusters)(raw_data, verbose) for raw_data in feats)
    else:
        clustering = [get_clusters(raw_data, verbose) for raw_data in feats]

    rep_data = np.vstack([sample_points_from_clustering(cdata["soft_labels"], raw_data, n) for cdata, raw_data in zip(clustering, feats)])
    
    logger.info(f"extracted ({rep_data.shape}) dataset from {len(feats)} animals, now clustering...")

    clustering = get_clusters(rep_data)
    return rep_data, clustering
    

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

        rep_data, clustering = cluster_for_strain(data, n=5000, verbose=True)
        return strain, rep_data, clustering
    
    bsoid = BSOID(config_file)
    feats = bsoid.load_features(collect=False)
    feats_id = ray.put(feats)

    logger.info(f"Processing {len(feats)} strains...")
    futures = [cluster_strain_data.remote(strain, feats_id) for strain in feats.keys()]
    
    pbar, results = tqdm(total=len(futures)), []
    while len(futures) > 0:
        n = len(futures) if len(futures) < num_cpus else num_cpus
        fin, rest = ray.wait(futures, num_returns=n)
        results.extend(ray.get(fin))
        futures = rest
        pbar.update(n)

    rep_data, clustering = {}, {}
    for res in results:
        strain, data, labels = res
        rep_data[strain] = data
        clustering[strain] = labels

    return rep_data, clustering

def collect_strainwise_labels(labels):
    def extract_label_info(strain, clusterer):
        soft_assignments = np.argmax(hdbscan.all_points_membership_vectors(clusterer), axis=1)
        return (strain, {"assignments": soft_assignments.astype("int"), "exemplars": clusterer.exemplars_indices_})

    from joblib import Parallel, delayed
    labels = Parallel(n_jobs=-1)(delayed(extract_label_info)(strain, clusterer) for strain, clusterer in labels.items())
    labels = {x[0]: x[1] for x in labels}
    return labels

def collect_strainwise_clusters(feats: dict, labels: dict, thresh: float, use_exemplars: bool):
    feats = collect_strainwise_feats(feats)
    labels = collect_strainwise_labels(labels)

    k, clusters = 0, {}
    for strain in feats.keys():
        class_labels, exemplars = labels[strain]["assignments"], labels[strain]["exemplars"]
        
        # threshold by entropy
        n = class_labels.max() + 1
        class_ids, counts = np.unique(class_labels, return_counts=True)
        prop = [x/class_labels.size for x in counts]
        entropy_ratio = -sum(p * np.log2(p) for p in prop) / max_entropy(n)

        if entropy_ratio >= thresh:
            logger.info(f"pooling {len(class_ids)} clusters from {strain} with entropy ratio {entropy_ratio}")
            for class_id in class_ids:
                if use_exemplars:
                    clusters[f"{strain}:{class_id}:{k}"] = feats[strain][exemplars[class_id],:]
                else:
                    clusters[f"{strain}:{class_id}:{k}"] = feats[strain][np.where(class_labels == class_id)[0]]
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

def pairwise_similarity(feats, labels, thresh):
    import ray
    import psutil

    clusters = collect_strainwise_clusters(feats, labels, thresh)
    del feats, labels
    
    logger.info(f"total clusters: {len(clusters)}")

    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)
    logger.info(f"running on: {num_cpus} CPUs")
    
    @ray.remote
    def par_pwise(idx1, idx2, clusters):
        sim = density_separation_similarity(idx1, idx2, clusters, metric="cosine") 
        idx1, idx2 = int(idx1.split(':')[-1]), int(idx2.split(':')[-1])
        return [sim, idx1, idx2]
    
    clusters_id = ray.put(clusters)
    pwise_combs = list(combinations(list(clusters.keys()), 2))
    N = len(pwise_combs)

    pbar, sim = tqdm(total=len(pwise_combs)), []
    
    k, futures = 0, []
    for _ in range(num_cpus):
        idx1, idx2 = pwise_combs[k]
        futures.append(par_pwise.remote(idx1, idx2, clusters_id))

    while k < len(pwise_combs):
        fin, futures = ray.wait(futures, num_returns=min(num_cpus, len(futures)), timeout=3000)
        sim.extend(ray.get(fin))
        pbar.update(len(fin))
        k += len(fin)
        
        for i in range(k, min(k+num_cpus, len(pwise_combs))):
            idx1, idx2 = pwise_combs[i]
            futures.append(par_pwise.remote(idx1, idx2, clusters_id))
    
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
    model = CatBoostClassifier(loss_function="MultiClass", eval_metric="Accuracy", iterations=3000, task_type="GPU", verbose=True)
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
    import logging
    logging.basicConfig(level=logging.INFO)

    bsoid = BSOID("./config/config.yaml")
    fdata = bsoid.load_filtered_data()["C57BL/6J"]
    for data in fdata:
        data['x'] = np.hstack((data['x'][:,:3].mean(axis=1).reshape(-1,1), data['x'][:,3:]))
        data['y'] = np.hstack((data['y'][:,:3].mean(axis=1).reshape(-1,1), data['y'][:,3:]))

    import psutil
    from joblib import Parallel, delayed
    feats = Parallel(n_jobs=psutil.cpu_count(logical=False))(delayed(extract_comb_feats)(data, bsoid.fps, None) for data in fdata)

    rep_data, clustering = cluster_for_strain(feats, 5000, parallel=True, verbose=True)
    with open("/fastscratch/laadd/strain.data", "wb") as f:
        joblib.dump([rep_data, clustering], f)