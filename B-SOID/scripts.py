import joblib
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from BSOID.bsoid import BSOID
from analysis import *

logging.basicConfig(level=logging.INFO, filename='training.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

GET_DATA          = False
PROCESS_CSVS      = False
LOAD_FROM_DATASET = True
GET_FEATURES      = False
UMAP_REDUCE       = False
CLUSTER_DATA      = False

def main(config_file, n=None, n_strains=None):
    bsoid = BSOID(config_file)

    if GET_DATA:
        bsoid.get_data(parallel=True)
    if PROCESS_CSVS:
        bsoid.process_csvs()
    if LOAD_FROM_DATASET:
        bsoid.load_from_dataset(n, n_strains)
    if GET_FEATURES:
        bsoid.features_from_points(parallel=True)
    if UMAP_REDUCE:
        bsoid.umap_reduce()
    if CLUSTER_DATA:
        bsoid.identify_clusters_from_umap()

def hyperparamter_tuning(config_file):
    bsoid = BSOID(config_file)
    bsoid.load_from_dataset(n=10)
    bsoid.features_from_points()

    n_nbrs = range(50, 201, 25)
    n_clusters = []
    for n in n_nbrs:
        print(f"n_nbrs: {n}")
        bsoid.umap_params["n_neighbors"] = n
        bsoid.umap_reduce()
        res = bsoid.identify_clusters_from_umap()
        n_clusters.append(np.unique(res[2]).max() + 1)

    print(n_clusters)


def validate_and_train(config_file):
    bsoid = BSOID(config_file)
    bsoid.validate_classifier()
    bsoid.train_classifier()    

def results(config_file):
    bsoid = BSOID(config_file)

    video_dir = bsoid.test_dir + '/videos'
    csv_dir = bsoid.test_dir
    bsoid.create_examples(csv_dir, video_dir, bout_length=3, n_examples=10)

def small_umap(config_file, outdir, n):
    bsoid = BSOID(config_file)
    bsoid.load_from_dataset(n)
    bsoid.features_from_points(parallel=True)
    
    umap_results = bsoid.umap_reduce()
    clustering_results = bsoid.identify_clusters_from_umap()
    
    import os
    with open(os.path.join(outdir, "2d_umap_results"), "wb") as f:
        joblib.dump(umap_results, f)
    with open(os.path.join(outdir, "2d_cluster_results"), "wb") as f:
        joblib.dump(clustering_results, f)

def ensemble_pipeline(config_file, outdir, subsample_size=int(2e5)):
    import hdbscan
    import umap
    from tqdm import tqdm
    from BSOID.bsoid import BSOID
    from BSOID.utils import cluster_with_hdbscan

    bsoid = BSOID(config_file)
    _, feats_sc = bsoid.load_features(collect=True)

    embed_maps, clusterers = [], []
    feats_sc = np.random.permutation(feats_sc)

    for i in tqdm(range(0, feats_sc.shape[0], subsample_size)):
        if i + subsample_size < feats_sc.shape[0]:
            subset = feats_sc[i:i+subsample_size,:]
        else:
            subset = feats_sc[i:]
        
        mapper = umap.UMAP(n_components=bsoid.reduced_dim, **bsoid.umap_params).fit(subset)
        _, _, _, clusterer = cluster_with_hdbscan(mapper.embedding_, bsoid.cluster_range, bsoid.hdbscan_params)

        embed_maps.append(mapper)
        clusterers.append(clusterer)
    
    with open(os.path.join(bsoid.base_dir + "ensemble_test_ckpt1.sav"), "wb") as f:
        joblib.dump([embed_maps, clusterers], f)
    
    from joblib import Parallel, delayed
    def transform_fn(clusterer, embed_map, X):
        embedding = embed_map.transform(X)
        labels = hdbscan.approximate_predict(clusterer, embedding)[0]
        return labels
    
    labels = Parallel(n_jobs=-1)(delayed(transform_fn)(clusterer, embed_mapper, feats_sc) for clusterer, embed_mapper in zip(clusterers, embed_maps))
    with open(os.path.join(outdir, "ensemble_clustering.sav"), "rb") as f:
        joblib.dump(labels, f)

def strainwise_test(config_file, outdir):
    from two_phase_clustering import strainwise_clustering

    bsoid = BSOID(config_file)
    bsoid.load_from_dataset(n=10)
    bsoid.features_from_points()

    import logging
    logging.basicConfig(
            level=logging.INFO, 
            filemode='a', 
            filename=os.path.join(outdir, "clustering-strainwise.log"), 
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S'
        )

    strainwise_clustering(config_file, outdir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("scripts.py")
    parser.add_argument("--config", type=str, help="configuration file for B-SOID")
    parser.add_argument("--script", type=str, help="script to run", choices=["results", "validate_and_train", "hyperparamter_tuning", "main", "small_umap", "ensemble_pipeline", "strainwise_test"])
    parser.add_argument("--n", type=int)
    parser.add_argument("--n_strains", type=int, default=None)
    parser.add_argument("--outdir", type=str)
    args = parser.parse_args()

    if args.script == "main":
        main(config_file=args.config, n=args.n, n_strains=args.n_strains)
    elif args.script == "small_umap":
        small_umap(config_file=args.config, outdir=args.outdir, n=args.n)
    elif args.script in ["ensemble_pipeline", "strainwise_test"]:
        ensemble_pipeline(config_file=args.config, outdir=args.outdir)
    else:
        eval(args.script)(args.config)
