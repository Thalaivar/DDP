from numpy.random import choice
from new_clustering import strain_pairs_sim
from sklearn.metrics import cluster
from BSOID.features import extract_temporal_feats
import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from BSOID.bsoid import BSOID
from analysis import *

import warnings
warnings.filterwarnings("ignore")

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
        bsoid.features_from_points()
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

def cluster_collect_embed(max_samples, thresh):
    import os
    import matplotlib.pyplot as plt
    from new_clustering import (
                                collect_strainwise_feats, 
                                collect_strainwise_clusters, 
                                reduce_data,
                            )   
    from BSOID.utils import cluster_with_hdbscan
    
    save_dir = "/home/laadd/data"
    with open(os.path.join(save_dir, "strainwise_labels.sav"), "rb") as f:
        feats, _,  labels = joblib.load(f)
    clusters = collect_strainwise_clusters(feats, labels, thresh)
    del feats, labels

    feats = []
    for _, data in clusters.items():
        if data["feats"].shape[0] > max_samples:
            feats.append(np.random.permutation(data["feats"])[:max_samples])
        else:
            feats.append(data["feats"])
    del clusters
    feats = np.vstack(feats)
    logging.info(f"running UMAP on: {feats.shape[0]}")
    
    embedding = reduce_data(feats, n_neighbors=60, n_components=13)
    results = cluster_with_hdbscan(embedding, [0.4, 1.2, 25], {"prediction_data": True, "min_samples": 1})
    
    from BSOID.utils import max_entropy
    prop = [p / results[2].size for p in np.unique(results[2], return_counts=True)[1]]
    entropy_ratio = -sum(p * np.log2(p) for p in prop) / max_entropy(results[0].max() + 1)

    logging.info(f"identified {results[0].max() + 1} clusters with entropy ratio {entropy_ratio}")

    groups = {}
    for lab in np.unique(results[2]):
        groups[lab] = feats[np.where(results[2] == lab)[0]]

    with open(os.path.join(save_dir, "cluster_collect_embed.sav"), "wb") as f:
        joblib.dump(groups, f)

def strainwise_cluster(config_file, save_dir, logfile):
    os.environ["NUMBA_NUM_THREADS"] = "1"
    from new_clustering import cluster_strainwise

    bsoid = BSOID(config_file)

    # bsoid.load_from_dataset(n=10)
    # bsoid.features_from_points()
 
    rep_data, clustering = cluster_strainwise(config_file, save_dir, logfile)
    
    with open(os.path.join(save_dir, "strainwise_labels.sav"), "wb") as f:
        joblib.dump([rep_data, clustering], f)

def rep_cluster(config_file, strain, save_dir, n):
    from new_clustering import cluster_for_strain
    from sklearn.ensemble import RandomForestClassifier

    bsoid = BSOID(config_file)
    feats = bsoid.load_features(collect=False)
    feats = feats[strain]

    Xtest = bsoid.get_random_animal_data(strain)
    failed = False

    labels = []
    for _ in range(n):
        rep_data, clustering = cluster_for_strain(feats, 5000, parallel=False, verbose=True)
        if rep_data is None:
            failed = True
            break
        model = RandomForestClassifier(n_jobs=-1)
        model.fit(rep_data, clustering["soft_labels"])
        labels.append(model.predict(Xtest))
    
    if not failed:
        with open(os.path.join(save_dir, f"{strain}.data"), "wb") as f:
            joblib.dump(np.array(labels), f)

def calculate_pairwise_similarity(save_dir, thresh, sim_measure):
    from new_clustering import strain_pairs_sim

    with open(os.path.join("/home/laadd/data", "strainwise_labels.sav"), "rb") as f:
        feats, clustering = joblib.load(f)

    sim = strain_pairs_sim(feats, clustering, thresh, sim_measure)
    sim.to_csv(os.path.join(save_dir, "pairwise_sim.csv"))

    # with open(os.path.join(save_dir, "pairwise_sim.sav"), "wb") as f:
    #     joblib.dump([sim, thresh], f)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser("scripts.py")
    parser.add_argument("--config", type=str, help="configuration file for B-SOID")
    parser.add_argument("--script", type=str, help="script to run", choices=[
                                                                    "results", 
                                                                    "validate_and_train", 
                                                                    "hyperparamter_tuning", 
                                                                    "main", 
                                                                    "small_umap", 
                                                                    "ensemble_pipeline", 
                                                                    "cluster_collect_embed", 
                                                                    "calculate_pairwise_similarity", 
                                                                    "strainwise_cluster",
                                                                    "rep_cluster"
                                                                ])
    parser.add_argument("--n", type=int)
    parser.add_argument("--n_strains", type=int, default=None)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--strain-file", type=str)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--thresh", type=float)
    parser.add_argument("--logfile", type=str)
    parser.add_argument("--strain", type=str)
    parser.add_argument("--sim-measure", type=str, choices=[
                                                        "density_separation_similarity", 
                                                        "dbcv_index_similarity", 
                                                        "roc_similarity",
                                                        "minimum_distance_similarity",
                                                        "hausdorff_similarity"
                                                    ]
                                                )
    args = parser.parse_args()

    import logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if args.save_dir is not None:
        logging.basicConfig(level=logging.INFO, filename=f"{args.save_dir}/{args.script}.log", filemode="w")
    else:
        logging.basicConfig(level=logging.INFO, filename=f"./{args.script}.log", filemode="w")

    if args.script == "main":
        main(config_file=args.config, n=args.n, n_strains=args.n_strains)
    elif args.script == "small_umap":
        small_umap(config_file=args.config, outdir=args.outdir, n=args.n)
    elif args.script == "ensemble_pipeline":
        eval(args.script)(config_file=args.config, outdir=args.outdir)
    elif args.script == "cluster_collect_embed":
        cluster_collect_embed(args.max_samples, args.thresh)
    elif args.script == "calculate_pairwise_similarity":
        calculate_pairwise_similarity(args.save_dir, args.thresh, args.sim_measure)
    elif args.script == "strainwise_cluster":
        strainwise_cluster(args.config, args.save_dir, args.logfile)
    elif args.script == "rep_cluster":
        rep_cluster(args.config, args.strain, args.save_dir, args.n)
    else:
        eval(args.script)(args.config)
