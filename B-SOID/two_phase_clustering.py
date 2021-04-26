import os
import umap
import joblib
import numpy as np

from tqdm import tqdm
from BSOID.bsoid import BSOID
from BSOID.utils import cluster_with_hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

strainwise_reduced_dim = 13
strainwise_n_neighbors = 60
strainwise_cluster_rng = [0.5, 1.0, 11]


def collect_strainwise_data(feats):
    for strain, data in feats.items():
        feats[strain] = np.vstack(data)
    return feats

def strainwise_clustering(config_file, outdir):
    bsoid = BSOID(config_file)
    feats = bsoid.load_features(collect=False)
    feats = collect_strainwise_data(feats)
    
    embedding, labels = {}, {}
    pbar = tqdm(total=len(feats))
    for strain, data in feats.items():
        embedding[strain] = reduce_data(data, n_components=12, n_neighbors=90)
        labels[strain] = cluster_with_hdbscan(embedding[strain], [0.4, 1.2, 25], bsoid.hdbscan_params)
        pbar.update(1)

    with open(os.path.join(outdir, "strainwise_labels.sav"), "rb") as f:
        joblib.dump([embedding, labels], f)

def reduce_data(feats: np.ndarray, n_neighbors: int, n_components: int):
    feats = StandardScaler().fit_transform(feats)

    mapper = umap.UMAP(
            n_components=n_components, 
            min_dist=0.0, 
            n_neighbors=n_neighbors
        ).fit(feats)
    return mapper.embedding_

def find_params(feats):
    import random
    import seaborn as sns
    import matplotlib.pyplot as plt

    strain = random.sample(list(feats.keys()), 1)[0]
    data = feats[strain]
    for strain, data in feats.items():
    # for n_neighbors in range(15, 106, 15):
        # for dim in range(4, 13, 4):
            # print(f"n_neighbors: {n_neighbors} ; dim: {dim}")
        logging.info(f"Data from: {strain}")
        embedding = reduce_data(data, n_neighbors=90, n_components=12)
        labels = cluster_with_hdbscan(embedding, [0.4, 1.0, 25], {"prediction_data": True, "min_samples": 1})[2]

if __name__ == "__main__":
    bsoid = BSOID("./config/config.yaml")
    feats = bsoid.load_features(collect=False)
    feats = collect_strainwise_data(feats)
    find_params(feats)