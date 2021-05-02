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
THRESH = 0.75

DISC_MODEL = LinearDiscriminantAnalysis(solver="svd")
CV = StratifiedKFold(n_splits=5, shuffle=True)

GROUPWISE_UMAP_PARAMS = {
    "n_neighbors": 75,
    "n_components": 3
}
GROUPWISE_CLUSTER_RNG = [1, 5, 25]

# CLF = RandomForestClassifier(class_weight="balanced", n_jobs=1)
# from catboost import CatBoostClassifier
# CLF = CatBoostClassifier(loss_function="MultiClass", eval_metric="Accuracy", iterations=10000, task_type="GPU", verbose=True)

from sklearn.neural_network import MLPClassifier
CLF = MLPClassifier(hidden_layer_sizes= [100, 10],activation= 'logistic',solver= 'adam',learning_rate= 'constant',learning_rate_init= 0.001,alpha= 0.0001,max_iter= 1000,early_stopping= False,verbose=1)

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
    k, clusters, strain2cluster = 0, {}, {}
    for strain in feats.keys():
        labels[strain] = labels[strain].astype(int)
        n = labels[strain].max() + 1
        prop = [x/labels[strain].size for x in np.unique(labels[strain], return_counts=True)[1]]
        entropy_ratio = sum(p * np.log2(p) for p in prop) / sum(p * np.log2(p) for p in [1/n for _ in range(n)])
        if entropy_ratio >= THRESH:
            strain2cluster[strain] = []
            for class_id in np.unique(labels[strain]):
                idx = np.where(labels[strain] == class_id)[0]
                clusters[k] = {
                    "feats": feats[strain][idx,:],
                    "embed": embedding[strain][idx,:]
                }
                strain2cluster[strain].append(k)
                k += 1
    
    return clusters, strain2cluster
    
# def collect_strainwise_clusters(feats: dict, labels: dict, embedding: dict):
#     k, clusters, strain2cluster = 0, {}, {strain: [] for strain in feats.keys()}
#     for strain in feats.keys():
#         for class_id in np.unique(labels[strain]):
#             idx = np.where(labels[strain] == class_id)[0]
#             if idx.size >= THRESH:
#                 clusters[k] = {
#                     "feats": feats[strain][idx,:],
#                     "embed": embedding[strain][idx,:]
#                 }
#                 strain2cluster[strain].append(k)
#                 k += 1
    
#     return clusters, strain2cluster

def cluster_similarity(cluster1, cluster2):
    X = [cluster1["embed"], cluster2["embed"]]
    y = [np.zeros((X[0].shape[0],)), np.ones((X[1].shape[0], ))]
    X, y = np.vstack(X), np.hstack(y)

    model = clone(DISC_MODEL)

    model.fit(X, y)
    y_pred = model.predict(X)
    score = roc_auc_score(y, y_pred)
    
    return score

def pairwise_similarity(feats, embedding, labels):
    import ray
    import psutil

    clusters, strain2clusters = collect_strainwise_clusters(feats, labels, embedding)
    print(f"Total clusters: {len(clusters)}")
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

def similarity_matrix(sim):
    n_clusters = int(sim[:,2:].max()) + 1
    mat = np.zeros((n_clusters, n_clusters))

    for i in range(sim.shape[0]):
        idx1, idx2 = sim[i,2:].astype("int")
        mat[idx1,idx2] = mat[idx2,idx1] = sim[i,0]
    
    return mat

def group_clusters(clusters, sim):
    mapper = umap.UMAP(min_dist=0.0, **GROUPWISE_UMAP_PARAMS).fit(similarity_matrix(sim))
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
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score
    print("f1_score: ", f1_score(y_test, preds, average="weighted"))
    print("roc_auc_score: ", roc_auc_score(y_test, model.predict_proba(X_test), average="weighted", multi_class="ovo"))
    print("accuracy: ", accuracy_score(y_test, preds))
    print("balanced_acc: ", balanced_accuracy_score(y_test, preds))
    
    return model

def main():
    import os
    
    save_dir = "/home/laadd/data"
    config_file = "./config/config.yaml"

    # bsoid = BSOID(config_file)
    # bsoid.load_from_dataset(n=10)
    # feats = bsoid.load_features(collect=False)

    # embedding, labels = cluster_strainwise(config_file, save_dir)

    # with open(os.path.join(save_dir, "strainwise_labels.sav"), "wb") as f:
    #     joblib.dump([feats, embedding, labels], f)

    with open(os.path.join(save_dir, "strainwise_labels.sav"), "rb") as f:
        feats, embedding, labels = joblib.load(f)

    sim, strain2clusters = pairwise_similarity(feats, embedding, labels)

    with open(os.path.join(save_dir, "pairwise_sim.sav"), "wb") as f:
        joblib.dump([sim, strain2clusters], f)

def run():
    import os

    # save_dir = "/home/laadd/data"
    save_dir = "../../data"
    with open(os.path.join(save_dir, "strainwise_labels.sav"), "rb") as f:
        feats, embedding, labels = joblib.load(f)

    clusters, _ = collect_strainwise_clusters(feats, labels, embedding)
    del feats, labels, embedding

    with open(os.path.join(save_dir, "pairwise_sim.sav"), "rb") as f:
        sim, _ = joblib.load(f)

    groups = group_clusters(clusters, sim)
    del clusters, sim

    train_classifier(groups)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # run()
    main()