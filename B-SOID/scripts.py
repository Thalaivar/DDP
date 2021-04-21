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
GET_FEATURES      = True
UMAP_REDUCE       = True
CLUSTER_DATA      = True

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
    bsoid.features_from_points(parallel=True)

    n_nbrs = range(50, 700, 50)
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
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("scripts.py")
    parser.add_argument("--config", type=str, help="configuration file for B-SOID")
    parser.add_argument("--script", type=str, help="script to run", choices=["results", "validate_and_train", "hyperparamter_tuning", "main", "small_umap"])
    parser.add_argument("--n", type=int)
    parser.add_argument("--n_strains", type=int)
    parser.add_argument("--outdir", type=str)
    args = parser.parse_args()

    if args.script == "main":
        main(config_file=args.config, n=args.n, n_strains=args.n_strains)
    elif args.script == "small_umap":
        small_umap(config_file=args.config, outdir=args.outdir, n=args.n)
    else:
        eval(args.script)(args.config)
