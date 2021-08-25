import umap
import hdbscan
import logging
import numpy as np

from collections import defaultdict
from behaviourPipeline.utils import calculate_entropy_ratio
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def embed_data(feats: np.ndarray, **umap_params):
    feats = StandardScaler().fit_transform(feats)
    embedding = umap.UMAP(**umap_params).fit_transform(feats)
    return embedding

def cluster_data(embedding, cluster_range, **hdbscan_params):    
    highest_entropy = -np.infty
    numulab, entropy = [], []

    logger.info(f"clustering {embedding.shape} with range {cluster_range}")
    
    if not isinstance(cluster_range, list):
        min_cluster_range = [cluster_range]
    if len(cluster_range) == 2:
        min_cluster_range = np.linspace(*cluster_range, 25)
    elif len(cluster_range) == 3:
        cluster_range[-1] = int(cluster_range[-1])
        min_cluster_range = np.linspace(*cluster_range)
        
    for min_c in min_cluster_range:
        trained_classifier = hdbscan.HDBSCAN(min_cluster_size=int(round(min_c * 0.01 * embedding.shape[0])),
                                            **hdbscan_params).fit(embedding)
        
        labels = trained_classifier.labels_
        numulab.append(labels.max() + 1)
        
        entropy.append(calculate_entropy_ratio(labels))

        logger.info(f"identified {numulab[-1]} clusters with min_sample_prop={round(min_c,2)} and entropy ratio={round(entropy[-1], 3)}")
        if entropy[-1] > highest_entropy:
            highest_entropy = entropy[-1]
            best_clf = trained_classifier

    assignments = best_clf.labels_
    soft_clusters = hdbscan.all_points_membership_vectors(best_clf)
    soft_assignments = np.argmax(soft_clusters, axis=1)

    return assignments, soft_clusters, soft_assignments, best_clf

def embed_and_cluster(feats: np.ndarray, hdbscan_params: dict, umap_params: dict, cluster_range=[0.4, 1.2], return_embedding=False):
    embedding = embed_data(feats, **umap_params)
    labels, _, soft_labels, clusterer = cluster_data(embedding, cluster_range, **hdbscan_params)
    exemplars = [feats[idxs] for idxs in clusterer.exemplars_indices_]
    
    logger.info(f"embedded {feats.shape} to {embedding.shape[1]}D with {labels.max() + 1} clusters and entropy ratio={round(calculate_entropy_ratio(soft_labels),3)}")
    if return_embedding:
        return embedding, {"labels": labels, "soft_labels": soft_labels, "exemplars": exemplars}
    else:
        return {"labels": labels, "soft_labels": soft_labels, "exemplars": exemplars}

def find_templates(labels, feats, num_points, use_inverse_density=True):
    if num_points >= feats.shape[0]: return feats, labels

    # if labels contain noise samples remove them
    feats, labels = feats[labels >= 0], labels[labels >= 0]

    classes, counts = np.unique(labels, return_counts=True)
    num_in_group = {classes[i]: counts[i] for i in range(classes.size)}

    if use_inverse_density:
        # probabilty of selecting a group is inversely proportional to its density
        props = {lab: 1 / (count / labels.size) for lab, count in num_in_group.items()}
        prop_sum = sum(prop for _, prop in  props.items())
        props = {lab: prop / prop_sum for lab, prop in props.items()}
    else:
        props = {lab: count / labels.size for lab, count in num_in_group.items()}

    # representative dataset of templates
    rep_labels = np.random.choice(classes, size=num_points, replace=True, p=[prop for _, prop in props.items()])
    rep_classes, rep_counts = np.unique(rep_labels, return_counts=True)

    # get indices of templates
    idx = []
    for i in range(rep_classes.size):
        rep_counts[i] = min(rep_counts[i], num_in_group[rep_classes[i]])
        class_idxs = np.where(labels == rep_classes[i])[0]
        idx.append(np.random.choice(class_idxs, size=rep_counts[i], replace=False))

    return feats[np.hstack(idx)], labels[np.hstack(idx)]

def cluster_for_strain(feats, num_points, umap_params, hdbscan_params, cluster_range=[0.4, 1.2]):
    # embed and cluster each animal
    clusters = [embed_and_cluster(
                    f, 
                    hdbscan_params, 
                    umap_params, 
                    cluster_range,
                ) for f in feats]

    # get representative dataset from each animal
    templates = np.vstack([find_templates(clusters[i]["labels"], f, num_points)[0] for i, f in enumerate(feats)])
    logger.info(f"extracted {templates.shape} dataset from {len(feats)} animals")

    # embed and cluster templates again
    clustering = embed_and_cluster(templates, hdbscan_params, umap_params, cluster_range)
    return templates, clustering

def filter_strain_clusters(feats: dict, clustering: dict, thresh: float, use_exemplars: bool) -> dict:
    clusters = defaultdict(list)
    for strain in feats.keys():
        # extract labels for strain
        class_labels = clustering[strain]["soft_labels"], 

        # threshold by entropy
        class_ids, _ = np.unique(class_labels, return_counts=True)
        entropy_ratio = calculate_entropy_ratio(class_labels)

        if entropy_ratio < thresh: continue
        
        logger.info(f"pooling {len(class_ids)} clusters from {strain} with entropy ratio {entropy_ratio}")
        for class_id in class_ids:
            if use_exemplars: class_data = clustering[strain]["exemplars"][class_id]
            else: class_data = feats[strain][np.where(class_labels == class_id)[0]]
            clusters[strain].append(class_data)

    return clusters

# def calculate_pairwise_similarity(templates, clustering, thresh, sim_measure):
#     assert sim_measure in SIMILARITY_MEASURES, f"similarity measure must be one of {SIMILARITY_MEASURES}"

#     sim_measure = eval(sim_measure)
#     clusters = collect_strain_clusters(templates, clustering, thresh, use_exemplars=False)
#     del templates, clustering

#     logger.info(f"total clusters: {sum(len(data) for _, data in clusters.items())}")

#     num_cpus = psutil.cpu_count(logical=False)
#     ray.init(num_cpus=num_cpus)
#     logger.info(f"running on: {num_cpus} CPUs")

#     combs = []
#     for strain1, strain2 in combinations(list(clusters.keys()), 2):
#         for i in range(len(clusters[strain1])):
#             for j in range(len(clusters[strain2])):
#                 combs.append(f"{strain1};{strain2};{i};{j}")

#         combs.extend([f"{strain1};{strain1};{i};{j}" for i, j in combinations(range(len(clusters[strain1])), 2)]) 
#         combs.extend([f"{strain2};{strain2};{i};{j}" for i, j in combinations(range(len(clusters[strain2])), 2)])
    
#     @ray.remote
#     def par_pwise(comb, clusters, sim_measure):
#         strain1, strain2, i, j = comb.split(';')
#         i, j = int(i), int(j)
#         X1 = clusters[strain1][i].copy()
#         X2 = clusters[strain2][j].copy()

#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             sim_val = sim_measure(X1, X2, metric="cosine")
        
#         return [sim_val, comb]
    
#     clusters_id = ray.put(clusters)
#     pbar, sim = tqdm(total=len(combs)), []
    
#     k, futures = 0, []
#     for k in range(num_cpus):
#         futures.append(par_pwise.remote(combs[k], clusters_id, sim_measure))

#     while k < len(combs):
#         fin, futures = ray.wait(futures, num_returns=min(num_cpus, len(futures)), timeout=3000)
#         sim.extend(ray.get(fin))
#         pbar.update(len(fin))
#         k += len(fin)
        
#         for i in range(k, min(k+num_cpus, len(combs))):
#             futures.append(par_pwise.remote(combs[i], clusters_id, sim_measure))

#     sim_data = {"sim": [], "strain1": [], "strain2": [], "idx1": [], "idx2": []}
#     for data in sim:
#         sim_data["sim"].append(data[0])
#         strain1, strain2, idx1, idx2 = data[1].split(';')
#         sim_data["strain1"].append(strain1)
#         sim_data["idx1"].append(int(idx1))
#         sim_data["strain2"].append(strain2)
#         sim_data["idx2"].append(int(idx2))
    
#     import pandas as pd
#     return pd.DataFrame.from_dict(sim_data)