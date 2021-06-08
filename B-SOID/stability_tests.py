import sys
sys.path.insert(0, "/home/laadd/DDP/B-SOID/")

import os
import ray
import yaml
import psutil
import shutil
from BSOID.bsoid import BSOID
from BSOID.clustering import *
from BSOID.features import extract_bsoid_feats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

def bsoid_cluster(embeddings,  verbose=False, **hdbscan_params):    
    max_classes = -np.infty
    numulab, entropy = [], []

    if "cluster_range" not in hdbscan_params:
        cluster_range = [0.4, 1.2]
    else:
        cluster_range = hdbscan_params["cluster_range"]
        del hdbscan_params["cluster_range"]

    if verbose:
        logger.info(f"Clustering {embeddings.shape} with range {cluster_range}")
    
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

        if verbose:
            logger.info(f"identified {numulab[-1]} clusters with min_sample_prop={round(min_c,2)} and entropy ratio={round(entropy[-1], 3)}")
        if numulab[-1] > max_classes:
            max_classes = numulab[-1]
            best_clf = trained_classifier

    assignments = best_clf.labels_
    soft_clusters = hdbscan.all_points_membership_vectors(best_clf)
    soft_assignments = np.argmax(soft_clusters, axis=1)

    return assignments, soft_clusters, soft_assignments, best_clf

def get_bsoid_clusters(feats: np.ndarray, hdbscan_params: dict, umap_params: dict):
    embedding = embed_data(feats, scale=True, **umap_params)
    labels, _, soft_labels, _ = bsoid_cluster(embedding, verbose=False, **hdbscan_params)

    logger.info(f"embedded {feats.shape} to {embedding.shape[1]}D with {labels.max() + 1} clusters and entropy ratio={round(calculate_entropy_ratio(soft_labels),3)}")
    return {"labels": labels, "soft_labels": soft_labels}

def bsoid_stability_test_train_model(config_file, run_id, base_dir, train_size=int(5e5)):
    fdata = BSOID(config_file).load_filtered_data()

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config["run_id"] = f"run-{run_id}"
    config["base_dir"] = base_dir
    bsoid = BSOID(config)

    with open(os.path.join(bsoid.output_dir, f"{bsoid.run_id}_filtered_data.sav"), "wb") as f:
        joblib.dump(fdata, f)
    del fdata

    fdata = bsoid.load_filtered_data()
    
    feats, strains = [], list(fdata.keys())
    for strain in strains:
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
    
    shutil.rmtree(bsoid.base_dir, ignore_errors=True)

def my_stability_test(nruns=50, test_size=int(1e5)):
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)
    
    bsoid = BSOID("./config/config.yaml")
    feats = []
    for _, data in bsoid.load_features(collect=False).items():
        feats.extend(data)
    feats = np.vstack(feats)

    def run_test(feats, bsoid):
        templates, clustering = bsoid.load_strainwise_clustering()

        thresh = 0.85
        clusters = collect_strain_clusters(templates, clustering, thresh, use_exemplars=True)
        del templates, clustering

        templates = np.vstack([np.vstack(data) for _, data in clusters.items()])

        umap_params = bsoid.umap_params
        hdbscan_params = bsoid.hdbscan_params

        clustering = get_clusters(templates, hdbscan_params, umap_params, bsoid.scale_before_umap, verbose=True)
        
        model = CatBoostClassifier(task_type="GPU", loss_function="MultiClass", )
