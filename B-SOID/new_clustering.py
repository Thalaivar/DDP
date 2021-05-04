import umap
import joblib
import numpy as np

from tqdm import tqdm
from sklearn import clone
from BSOID.bsoid import BSOID
from BSOID.utils import max_entropy
from itertools import combinations
from sklearn.metrics import roc_auc_score
from BSOID.utils import cluster_with_hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# CLF = RandomForestClassifier(class_weight="balanced", n_jobs=1)
# from catboost import CatBoostClassifier
# CLF = CatBoostClassifier(loss_function="MultiClassOneVsAll", eval_metric="Accuracy", iterations=10000, task_type="GPU", verbose=True)

def reduce_data(feats: np.ndarray, **umap_params):
    feats = StandardScaler().fit_transform(feats)
    mapper = umap.UMAP(min_dist=0.0, **umap_params).fit(feats)
    return mapper.embedding_

def collect_strainwise_feats(feats: dict):
    for strain, animal_data in feats.items():
        feats[strain] = np.vstack(animal_data)
    return feats

def cluster_strainwise(config_file, save_dir):
    import ray
    import psutil

    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)

    @ray.remote
    def cluster_strain_data(strain, feats):
        logger = open("/home/laadd/DDP/B-SOID/strainwise_clustering.log", "a")
        data = feats[strain]

        logger.write(f"running for strain: {strain} with samples: {data.shape}\n")
        logger.addHandler(logging.StreamHandler())

        strainwise_umap_params = {"n_neighbors": 90, "n_components": 12}
        strainwise_cluster_rng = [0.4, 1.2, 25]
        hdbscan_params = {"prediction_data": True, "min_samples": 1, "core_dist_n_jobs": 1}
        
        embedding = reduce_data(data, **strainwise_umap_params)
        assignments, _, soft_assignments, _ = cluster_with_hdbscan(embedding, strainwise_cluster_rng, hdbscan_params)
        
        prop = [p / soft_assignments.size for p in np.unique(soft_assignments, return_counts=True)[1]]
        entropy_ratio = sum(p * np.log2(p) for p in prop) / max_entropy(assignments.max() + 1)

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
    for strain in embedding.keys():
        assignments, soft_assignments = labels[strain]
        
        # feats[strain] = feats[strain][assignments >= 0]
        # embedding[strain] = embedding[strain][assignments >= 0]
        # labels[strain] = soft_assignments[assignments >= 0]

        labels[strain] = soft_assignments
        logger.info(f"Strain: {strain} ; Features: {feats[strain].shape} ; Embedding: {embedding[strain].shape} ; Labels: {labels[strain].shape}")
    
    return feats, embedding, labels

def collect_strainwise_clusters(feats: dict, labels: dict, embedding: dict, thresh: float):
    feats, embedding, labels = collect_strainwise_labels(feats, embedding, labels)

    k, clusters, strain2cluster = 0, {}, {}
    for strain in feats.keys():
        labels[strain] = labels[strain].astype(int)

        # threshold by entropy
        n, class_ids, counts = labels[strain].max() + 1, np.unique(labels[strain], return_counts=True)
        prop = [x/labels[strain].size for x in counts]
        entropy_ratio = sum(p * np.log2(p) for p in prop) / max_entropy(n)

        if entropy_ratio >= thresh:
            strain2cluster[strain] = []
            for class_id in class_ids:
                idx = np.where(labels[strain] == class_id)[0]
                clusters[k] = {
                    "feats": feats[strain][idx,:],
                    "embed": embedding[strain][idx,:]
                }
                strain2cluster[strain].append(k)
                k += 1
    
    assert len(clusters) == sum(lab.max() + 1 for _, lab in labels.items())
    return clusters, strain2cluster

def cluster_similarity(cluster1, cluster2):
    X = [cluster1["feats"], cluster2["feats"]]
    y = [np.zeros((X[0].shape[0],)), np.ones((X[1].shape[0], ))]
    X, y = np.vstack(X), np.hstack(y)

    model = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    model.fit(X, y)

    Xproj = model.transform(X)
    y_preds = (Xproj < 0).astype(int)

    return roc_auc_score(y, y_preds)

def pairwise_similarity(feats, embedding, labels, thresh):
    import ray
    import psutil

    clusters, strain2clusters = collect_strainwise_clusters(feats, labels, embedding, thresh)
    print(f"Total clusters: {len(clusters)}")
    del feats, embedding, labels

    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)

    @ray.remote
    def par_pwise(idx1, idx2, clusters):
        sim = cluster_similarity(clusters[idx1], clusters[idx2])
        return [sim, idx1, idx2]
    
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

def impute_same_strain_values(sim, strain2cluster):
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
    
    same_strain_sim = filler.predict(sim[same_strain_idx, 1:])
    same_strain_sim = np.hstack((same_strain_sim.reshape(-1,1), sim[same_strain_idx,1:]))

    sim = np.vstack((diff_strain_sim, same_strain_sim))
    return sim

def similarity_matrix(sim, strain2cluster):
    sim = impute_same_strain_values(sim, strain2cluster)

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
    
    model = clone(CLF)
    model.set_params(**params)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, stratify=y)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    # preds = model.predict(X_test)

    # from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score
    # print("f1_score: ", f1_score(y_test, preds, average="weighted"))
    # print("roc_auc_score: ", roc_auc_score(y_test, model.predict_proba(X_test), average="weighted", multi_class="ovo"))
    # print("accuracy: ", accuracy_score(y_test, preds))
    # print("balanced_acc: ", balanced_accuracy_score(y_test, preds))
    
    return model

def run():
    import os

    # save_dir = "/home/laadd/data"
    save_dir = "../../data/2clustering"
    with open(os.path.join(save_dir, "strainwise_labels.sav"), "rb") as f:
        feats, embedding, labels = joblib.load(f)

    clusters, _ = collect_strainwise_clusters(feats, labels, embedding)
    del feats, labels, embedding

    with open(os.path.join(save_dir, "pairwise_sim.sav"), "rb") as f:
        sim, strain2cluster = joblib.load(f)

    groups = group_clusters(clusters, sim, strain2cluster)
    del clusters, sim

    train_classifier(groups)