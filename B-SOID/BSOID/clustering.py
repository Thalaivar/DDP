import logging
import hdbscan
from  _heapq import heapify
import numpy as np
from tqdm import tqdm
from CURE.cure import cure_cluster, CURE

class bigCURE(CURE):
    def __init__(self, desired_clusters, n_parts, clusters_per_part, n_rep, alpha):
        super().__init__(desired_clusters, n_rep, alpha)
        self.n_parts = n_parts
        self.k_per_part = clusters_per_part
        
    def init_w_clusters(self, clusters):
        data = []
        i = 0
        for c in clusters:
            data.extend(c.rep)
            c.rep_idx = [i + j for j in range(len(c.rep))]
            i += len(c.rep)
        data = np.vstack(data)
        self._create_kdtree(data)

        self.heap_ = clusters
        for c in self.heap_:
            closest_rep_idx, min_dist = self._closest_cluster(c, np.inf)
            c.closest = self._find_cluster_with_rep_idx(closest_rep_idx)
            c.distance = min_dist
        
        heapify(self.heap_)

def partition_dataset(data, n_parts):
    partitions = []
    batch_sz = data.shape[0] // n_parts
    
    logging.info('creating {} partitions of {} samples for batch size {}'.format(n_parts, data.shape[0], batch_sz))

    i = 0
    for _ in range(n_parts):
        if i + batch_sz < data.shape[0] - 1:
            partitions.append(data[i:i+batch_sz,:])
            i += batch_sz
        else:
            partitions.append(data[i:,:])
    
    return partitions
        
def preclustering(data, n_parts, min_clusters, max_clusters):
    parts = partition_dataset(data, n_parts)

    assignments = []
    logging.info(f'preclustering {len(parts)} partitions of data to have {max_clusters} clusters each')

    min_prop = 0.1
    for i in tqdm(range(len(parts))):
        part_assgn, min_prop = cluster_with_hdbscan(parts[i], min_k_per_part, max_clusters, min_prop)
        assignments.append(part_assgn)

    return parts, assignments
        
def cluster_with_hdbscan(data, min_k, max_k, min_prop=None):
    min_prop = 0.1 if min_prop is None else min_prop

    while True:
        min_cluster_size = int(round(min_prop * 0.01 * data.shape[0]))
        clusterer = hdbscan.HDBSCAN(min_cluster_size, min_samples=10, prediction_data=True).fit(data)
        soft_assignments = np.argmax(hdbscan.all_points_membership_vectors(clusterer), axis=1)
        n_clusters = len(np.unique(soft_assignments))
        if n_clusters <= max_k and n_clusters >= min_k:
            break
        elif n_clusters > max_k:
            logging.debug(f'clusters: {n_clusters} ; max clusters: {max_k} - increasing `min_cluster_size`')
            min_prop *= 1.2
        elif n_clusters < min_k:
            logging.debug(f'clusters: {n_clusters} ; min clusters: {min_k} - decreasing `min_cluster_size`')
            min_prop *= 0.8
        
    logging.debug(f'identified {n_clusters} clusters')
    return soft_assignments, min_prop

def clusters_from_assignments(partitions, assignments, **cluster_args):
    clusters = []

    logging.debug(f'creating clusters from HDBSCAN assignments for {len(partitions)} partitions')
    start_idx = 0
    for i in range(len(partitions)):
        for l in np.unique(assignments[i]):
            idx = np.where(assignments[i] == l)[0]
            data = partitions[i][idx]
            idx += start_idx
            clusters.append(cure_cluster(data, idx, **cluster_args))
        start_idx += partitions[i].shape[0]
    
    for i in tqdm(range(len(clusters))):
        clusters[i]._calculate_mean()
        clusters[i]._exemplars_from_data()

    return clusters