import os
import joblib

from tqdm import tqdm
from BSOID.bsoid import BSOID
from itertools import combinations

BASE_DIR = '/home/dhruvlaad/data'
RUN_ID = 'dis'

def generate_umap_embeddings(n: int, output_dir: str):
    bsoid = BSOID.load_config(BASE_DIR, RUN_ID)

    def save_random_embedding(i):
        results = bsoid.umap_reduce_all(reduced_dim=3, sample_size=int(7e5))
        with open(output_dir + f'/{i}_umap.sav', 'wb') as f:
            joblib.dump(results, f)
    
    from joblib import Parallel, delayed
    Parallel(n_jobs=-1)(delayed(save_random_embedding)(i) for i in range(n))

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
    files = [f'{data_dir}/{f}' for f in os.listdir(data_dir) if f.endswith('clusters.sav')]


def compare_two_clusterings(clustering_1, clustering_2):
    feats_1, labels_1 = clustering_1
    feats_2, labels_2 = clustering_2

if __name__ == "__main__":
    output_dir = '/mnt/tmp'
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    n = 10
    generate_umap_embeddings(n, output_dir)