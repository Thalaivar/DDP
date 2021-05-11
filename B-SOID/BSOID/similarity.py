from hdbscan.validity import validity_index
from scipy.spatial.distance import cdist

def metric_similarity(idx1, idx2, clusters, metric):
    X1, X2 = clusters[idx1], clusters[idx2]