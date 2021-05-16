from hdbscan.validity import *
from itertools import combinations
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist, directed_hausdorff
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def roc_similiarity(idx1, idx2, clusters, metric=None):
    X1, X2 = clusters[idx1], clusters[idx2]
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros((X1.shape[0],)), np.ones((X2.shape[0],)))).astype("int")

    model = LinearDiscriminantAnalysis(n_components=1)
    model.fit(X, y)
    Xproj = model.transform(X)
    
    return roc_auc_score(y, Xproj)

def hausdorff_similarity(idx1, idx2, clusters, metric=None):
    X1, X2 = clusters[idx1], clusters[idx2]
    dist = directed_hausdorff(X1, X2)[0]
    return dist