import os
from posixpath import join
from re import template
import joblib
import numpy as np
from ray._private import services
from analysis import *
from BSOID.bsoid import BSOID
from BSOID.clustering import *

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

def rooted_final_clustering(config_file, thresh, save_dir):
    bsoid = BSOID(config_file)
    templates, clustering = bsoid.load_strainwise_clustering()
    clusters = collect_strain_clusters(templates, clustering, thresh, use_exemplars=True)

    # retain 10% of each cluster as "roots"
    k, templates, labels = 0, [], []
    for _, data in clusters.items():
        for d in data:
            templates.append(d)
            cluster_labels = -1 * np.ones((d.shape[0], ))

            root_idxs = np.random.permutation(np.arange(cluster_labels.size))[:np.floor(0.1 * cluster_labels.size).astype(int)]
            cluster_labels[root_idxs] = k
            labels.append(cluster_labels)

            k += 1

    templates, labels = np.vstack(templates), np.hstack(labels)
    
    umap_params = bsoid.umap_params
    hdbscan_params = bsoid.hdbscan_params

    logger.info(f"running rooted embedding with {templates.shape} samples")

    if bsoid.scale_before_umapale:
        templates = StandardScaler().fit_transform(templates)
    
    embedding = umap.UMAP(**umap_params).fit_transform(templates, y=labels)
    labels, _, soft_labels, clusterer = cluster_with_hdbscan(embedding, verbose=True, **hdbscan_params)
    exemplars = [templates[idxs] for idxs in clusterer.exemplars_indices_]
    
    logger.info(f"embedded {templates.shape} to {embedding.shape[1]}D with {labels.max() + 1} clusters and entropy ratio={round(calculate_entropy_ratio(soft_labels),3)}")
    clustering = {"labels": labels, "soft_labels": soft_labels, "exemplars": exemplars}
    
    with open(os.path.join(save_dir, "rooted_together.data"), "wb") as f:
        joblib.dump([templates, clustering, thresh], f)

def cluster_collect_embed(config_file, thresh, save_dir):
    bsoid = BSOID(config_file)
    templates, clustering = bsoid.load_strainwise_clustering()

    if bsoid.training_set_size > 0:
        # subsample data
        num_points = np.floor(bsoid.training_set_size / len(templates)).astype(int)
        logger.info(f"generating training set with {num_points} per strain")
        for strain in templates.keys():
            templates[strain], clustering[strain]["soft_labels"] = find_templates(clustering[strain]["soft_labels"], templates[strain], num_points, use_inverse_density=False)

    clusters = collect_strain_clusters(templates, clustering, thresh, use_exemplars=False)
    del templates, clustering

    templates = np.vstack([np.vstack(data) for _, data in clusters.items()])
    logger.info(f"embedding {templates.shape} templates from {sum(len(data) for _, data in clusters.items())} clusters")

    umap_params = bsoid.umap_params
    hdbscan_params = bsoid.hdbscan_params

    clustering = get_clusters(templates, hdbscan_params, umap_params, bsoid.scale_before_umap, verbose=True)
    
    with open(os.path.join(save_dir, "together.data"), "wb") as f:
        joblib.dump([templates, clustering, thresh], f)


def strainwise_cluster(config_file, logfile):
    bsoid = BSOID(config_file)
    
    # bsoid.load_from_dataset(n=10)
    # bsoid.features_from_points()
    
    bsoid.cluster_strainwise(logfile)

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
                                                                    "hyperparamter_tuning", 
                                                                    "main", 
                                                                    "cluster_collect_embed", 
                                                                    "calculate_pairwise_similarity", 
                                                                    "strainwise_cluster",
                                                                    "rep_cluster",
                                                                    "rooted_final_clustering"
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
                                                        "roc_similiarity",
                                                        "minimum_distance_similarity",
                                                        "hausdorff_similarity"
                                                    ]
                                                )
    args = parser.parse_args()

    import logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # if args.save_dir is not None:
    #     logging.basicConfig(level=logging.INFO, filename=f"{args.save_dir}/{args.script}.log", filemode="w")
    # else:
    #     logging.basicConfig(level=logging.INFO, filename=f"./{args.script}.log", filemode="w")
    logging.basicConfig(level=logging.INFO)

    if args.script == "main":
        main(config_file=args.config, n=args.n, n_strains=args.n_strains)
    elif args.script == "small_umap":
        small_umap(config_file=args.config, outdir=args.outdir, n=args.n)
    elif args.script == "ensemble_pipeline":
        eval(args.script)(config_file=args.config, outdir=args.outdir)
    elif args.script == "cluster_collect_embed":
        cluster_collect_embed(args.config, args.thresh, args.save_dir)
    elif args.script == "rooted_final_clustering":
        rooted_final_clustering(args.config, args.thresh, args.save_dir)
    elif args.script == "calculate_pairwise_similarity":
        calculate_pairwise_similarity(args.save_dir, args.thresh, args.sim_measure)
    elif args.script == "strainwise_cluster":
        strainwise_cluster(args.config, args.logfile)
    elif args.script == "rep_cluster":
        rep_cluster(args.config, args.strain, args.save_dir, args.n)
    else:
        eval(args.script)(args.config)
