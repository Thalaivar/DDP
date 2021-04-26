import os
import umap
import joblib
import numpy as np

from BSOID.bsoid import BSOID
from BSOID.utils import cluster_with_hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

strainwise_reduced_dim = 13
strainwise_n_neighbors = 60
strainwise_cluster_rng = [0.5, 1.0, 11]

def strainwise_clustering(config_file, outdir):
    bsoid = BSOID(config_file)

    feats = bsoid.load_features(collect=False)
    embedding = {name: reduce_data(data) for name, data in feats.items()}

    clustering = {name: cluster_with_hdbscan(data, strainwise_cluster_rng, bsoid.hdbscan_params)[0]}
    
    coll_feats, coll_embedding, coll_data = {}, {}, {}
    
    k, strain2idx = 0, {"strain": [], "class_id": []}
    for strain in feats.keys():
        for idx in np.unique()

    with open(os.path.join(outdir, "./strainwise_clustering"), "wb") as f:
        joblib.dump([embedding, clustering], f)
    

    return embedding, clustering

def reduce_data(feats: list, max_sample_size=int(2e5)):
    feats = StandardScaler().fit_transform(np.vstack(feats))
    
    if feats.shape[0] > max_sample_size:
        feats = np.random.permutation(feats)[:max_sample_size]

    mapper = umap.UMAP(
            n_components=strainwise_reduced_dim, 
            min_dist=0.0, 
            n_neighbors=strainwise_n_neighbors
        ).fit(feats)
    return mapper.embedding_