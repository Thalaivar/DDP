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

    n_nbrs = [50, 60, 70, 100, 150, 200, 250, 300]
    for n in n_nbrs:
        bsoid.umap_params["n_neighbors"] = n
        bsoid.umap_reduce()
        bsoid.identify_clusters_from_umap()


def validate_and_train(config_file):
    bsoid = BSOID(config_file)
    bsoid.validate_classifier()
    bsoid.train_classifier()    

def results(config_file):
    bsoid = BSOID(config_file)

    video_dir = bsoid.test_dir + '/videos'
    csv_dir = bsoid.test_dir
    bsoid.create_examples(csv_dir, video_dir, bout_length=3, n_examples=10)

if __name__ == "__main__":
    # main(config_file="./config.yaml", n=10)
    hyperparamter_tuning(config_file="./config.yaml")
