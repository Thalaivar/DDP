import os
import joblib
import numpy as np

from tqdm import tqdm
from BSOID.bsoid import BSOID
from itertools import combinations

import logging
logging.basicConfig(level=logging.INFO)

BASE_DIR = '/home/dhruvlaad/data'
RUN_ID = 'dis'

def generate_umap_embeddings(n: int, output_dir: str):
    bsoid = BSOID.load_config(BASE_DIR, RUN_ID)

    def save_random_embedding(i):
        results = bsoid.umap_reduce(reduced_dim=3, sample_size=int(7e5))
        with open(output_dir + f'/{i}_umap.sav', 'wb') as f:
            joblib.dump(results, f)
    
    from joblib import Parallel, delayed
    Parallel(n_jobs=3)(delayed(save_random_embedding)(i) for i in range(n))

def cluster_embeddings(data_dir, cluster_range=[0.4,1.2]):
    bsoid = BSOID.load_config(BASE_DIR, RUN_ID)

    files = [f for f in os.listdir(data_dir) if f.endswith('umap.sav')]
    for i, f in enumerate(files):
        import shutil
        shutil.copyfile(f'{data_dir}/{f}', f'{bsoid.output_dir}/{bsoid.run_id}_umap.sav')

        results = bsoid.identify_clusters_from_umap(cluster_range)
        with open(f'{data_dir}/{i}_clusters.sav', 'wb') as datafile:
            joblib.dump(results, datafile)


def correlation_similarity(data_dir):
    feats_files = [f'{data_dir}/{f}' for f in os.listdir(data_dir) if f.endswith('umap.sav')]
    cluster_files = [f'{data_dir}/{f}' for f in os.listdir(data_dir) if f.endswith('clusters.sav')]

    feats_files.sort()
    cluster_files.sort()

    assert len(feats_files) == len(cluster_files)
    
    def compare(i, j, feats_files, cluster_files):
        with open(feats_files[i], 'rb') as f:
            _, feats_1, _ = joblib.load(f)
        with open(cluster_files[i], 'rb') as f:
            _, _, labels_1 = joblib.load(f)
        with open(feats_files[j], 'rb') as f:
            _, feats_2, _ = joblib.load(f)
        with open(cluster_files[j], 'rb') as f:
            _, _, labels_2 = joblib.load(f)
        
        return compare_two_clusterings([feats_1, labels_1], [feats_2, labels_2])
    
    from joblib import Parallel, delayed
    correlations = Parallel(n_jobs=-1)(delayed(compare)(i, j, feats_files, cluster_files) for i, j in combinations(range(len(feats_files)), 2))
    
    return correlations

def compare_two_clusterings(clustering_1, clustering_2):
    feats_1, labels_1 = clustering_1
    feats_2, labels_2 = clustering_2

    assert len(feats_1) == len(feats_2)
    intersect_labels_1, intersect_labels_2 = [], []
    n_intersect = 0
        
    for i in tqdm(range(len(feats_1))):
        for j in range(len(feats_2)):
            if np.allclose(feats_1[i], feats_2[j]):
                intersect_labels_1.append(labels_1[i])
                intersect_labels_2.append(labels_2[j])
                n_intersect += 1

    logging.info(f'found {n_intersect} common samples in the two datasets')

    correlation_dot_prod = 0
    for i, j in combinations(range(n_intersect), 2):
        correlation_dot_prod += (intersect_labels_1[i] == intersect_labels_1[j])*(intersect_labels_2[i] == intersect_labels_2[j])
    
    return correlation_dot_prod
    

def stability_by_classification(feats, labels):
    feats_train, feats_test = feats
    labels_train, labels_test = labels

    from sklearn.neural_network import MLPClassifier
    MLP_PARAMS = {
        'hidden_layer_sizes': (128, 64),  # 100 units, 10 layers
        'activation': 'logistic',  # logistics appears to outperform tanh and relu
        'solver': 'adam',
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,  # learning rate not too high
        'alpha': 0.0001,  # regularization default is better than higher values.
        'max_iter': 1000,
        'early_stopping': False,
        'verbose': 0  # set to 1 for tuning your feedforward neural network
    }
    model = MLPClassifier(**MLP_PARAMS).fit(feats_train, labels_train)
    preds = model.fit(feats_test)

    def calculate_minimum_dist(labels_test, preds):
        assert len(labels_test) == len(preds)
        n = len(labels_test)

        dist = 0
        for i in range(n):
            dist += 1*(preds[i] == labels_test[i])
        
        return dist / n
    
    def premute_labels(labels):
        original_labels = np.unique(labels)
        perm_labels = np.random.permutation(original_labels)

        for i in range(len(labels)):
            labels[i] = perm_labels[labels[i]]
        
        return labels

if __name__ == "__main__":
    output_dir = '/mnt/tmp'
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    n = 10
    generate_umap_embeddings(n, output_dir)