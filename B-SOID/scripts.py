import joblib
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from BSOID.bsoid import BSOID

logging.basicConfig(level=logging.INFO)


# model_load_params = None
MODEL_LOAD_PARAMS = {'run_id': 'dis', 'base_dir': '/home/laadd/data'}

def main(model_load_params=None, get_data=False, preprocess=False, extract_features=False, embed=True, cluster=False):
    if model_load_params is None:
        bsoid_params = {'run_id': 'dis',
                        'base_dir': '/home/dhruvlaad/data',
                        'fps': 30,
                        'stride_window': 3,
                        'conf_threshold': 0.3
            }
        bsoid = BSOID(**bsoid_params)
        bsoid.save()
    else:   
        bsoid = BSOID.load_config(**model_load_params)

    if get_data:
        bsoid.get_data(parallel=True)
    if preprocess:
        bsoid.process_csvs()
    if extract_features:
        bsoid.features_from_points(parallel=True)
    if embed:
        # bsoid.best_reduced_dim()
        # reduced_dim = int(input('Enter reduced dimensions for embedding: '))
        bsoid.umap_reduce(reduced_dim=3, sample_size=-1)
    if cluster:
        bsoid.identify_clusters_from_umap(cluster_range=[0.2, 1.0, 9])

def get_cluster_information(run_id='dis', base_dir='D:/IIT/DDP/data'):
    bsoid = BSOID.load_config(run_id=run_id, base_dir=base_dir)

    assignments, soft_clusters, soft_assignments, clusterer = bsoid.load_identified_clusters()
    assignments = assignments.astype(np.int8)
    
    prop = [0 for _ in range(assignments.max() + 1)]  
    for idx in assignments:
        if idx >= 0:
            prop[idx] += 1
    prop = np.array(prop)
    prop = prop/prop.sum()
    sns.barplot(x=np.arange(assignments.max() + 1), y=prop)
    plt.show()

def validate_and_train(run_id='dis', base_dir='D:/IIT/DDP/data'):
    bsoid = BSOID.load_config(run_id=run_id, base_dir=base_dir)
    bsoid.validate_classifier()
    bsoid.train_classifier()

def results(run_id='dis', base_dir='D:/IIT/DDP/data'):
    bsoid = BSOID.load_config(run_id=run_id, base_dir=base_dir)

    video_dir = bsoid.test_dir + '/videos'
    csv_dir = bsoid.test_dir
    bsoid.create_examples(csv_dir, video_dir, bout_length=3, n_examples=10)

if __name__ == "__main__":
    main(model_load_params=MODEL_LOAD_PARAMS, preprocess=True, extract_features=True, embed=True, cluster=False)
    # results()
    # get_cluster_information()