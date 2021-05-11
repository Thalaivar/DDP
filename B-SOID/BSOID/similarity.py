from itertools import combinations
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from hdbscan.validity import validity_index, internal_minimum_spanning_tree

def dbcv_index_similarity(idx1, idx2, clusters, metric):
    X1, X2 = clusters[idx1], clusters[idx2]
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros((X1.shape[0],)), np.ones((X2.shape[0],))))
    return validity_index(X, y, metric)

def minimum_distance_similarity(idx1, idx2, clusters, metric):
    X1, X2 = clusters[idx1], clusters[idx2]
    min_dist = cdist(X1, X2, metric=metric).min()
    return min_dist

def all_points_core_distance(X):
    n, d = X.shape
    dist_mat = pairwise_distances(X)

    core_dist = np.zeros((n,))
    for i in range(n):
        dist = 0.0
        for j in range(n):
            if i != j:
                i, j = j, i if i > j else i, j
                dist += (1 / dists[n * i + j - ((i + 2) * (i + 1)) // 2]) ** d
        dist /= (n - 1)
        core_dist[i] = dist ** (-1 / d)    
    
    for i, j in combinations(range(n), 2):
        i, j = j, i if i > j else i, j
        dists[n * i + j - ((i + 2) * (i + 1)) // 2] = max(core_dist[i], core_dist[j], dists[n * i + j - ((i + 2) * (i + 1)) // 2])
    
    nodes, edges = internal_minimum_spanning_tree(dists)
    return core_dist, nodes, edges
    
def density_separation_similarity(idx1, idx2, clusters, metric):
    X1, X2 = clusters[idx1], clusters[idx2]

    core_dist_1, internal_nodes_1, internal_edges_1 = all_points_core_distance(X) 
