from itertools import combinations
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from hdbscan.validity import *

def dbcv_index_similarity(idx1, idx2, clusters, metric):
    X1, X2 = clusters[idx1], clusters[idx2]
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros((X1.shape[0],)), np.ones((X2.shape[0],))))
    return validity_index(X, y, metric)

def minimum_distance_similarity(idx1, idx2, clusters, metric):
    X1, X2 = clusters[idx1], clusters[idx2]
    min_dist = cdist(X1, X2, metric=metric).min()
    return min_dist
    
def density_separation_similarity(idx1, idx2, clusters, metric):
    X1, X2 = clusters[idx1], clusters[idx2]
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros((X1.shape[0],)), np.ones((X2.shape[0],))))

    core_distances, internal_nodes, internal_edges = [], [], []
    for i in range(2):
        mr_distances, core_distances[i] = all_points_mutual_reachability(X, y, i, metric)
        internal_nodes[i], internal_edges[i] = internal_minimum_spanning_tree(mr_distances)

    density_sep = np.inf * np.ones((max_cluster_id, max_cluster_id), dtype=np.float64)
    for i in range(2):
        if np.sum(y == i) == 0:
            continue

        internal_nodes_i = internal_nodes[i]
        
