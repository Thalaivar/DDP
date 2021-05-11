from itertools import combinations
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from hdbscan.validity import *

def dbcv_index_similarity(idx1, idx2, clusters, metric):
    X1, X2 = clusters[idx1], clusters[idx2]
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros((X1.shape[0],)), np.ones((X2.shape[0],)))).astype("int")
    return validity_index(X, y, metric)

def minimum_distance_similarity(idx1, idx2, clusters, metric):
    X1, X2 = clusters[idx1], clusters[idx2]
    min_dist = cdist(X1, X2, metric=metric).min()
    return min_dist
    
def density_separation_similarity(idx1, idx2, clusters, metric):
    X1, X2 = clusters[idx1], clusters[idx2]
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros((X1.shape[0],)), np.ones((X2.shape[0],)))).astype("int")

    core_distances, internal_nodes = {}, {}
    for i in range(2):
        mr_distances, core_distances[i] = all_points_mutual_reachability(X, y, i, metric)
        internal_nodes[i], _ = internal_minimum_spanning_tree(mr_distances)

    density_sep = density_separation(X, y, 0, 1, internal_nodes[0], internal_nodes[1], core_distances[0], core_distances[1])
    return density_sep