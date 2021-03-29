import joblib
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from BSOID.bsoid import BSOID
from analysis import *

logging.basicConfig(level=logging.INFO, filename='training.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# model_load_params = None
MODEL_LOAD_PARAMS = {'run_id': 'dis', 'base_dir': '/home/laadd/data'}

def main():
    stride_window = round(350 * 30 / 1000)
    bsoid_params = MODEL_LOAD_PARAMS
    bsoid_params['fps'] = FPS
    bsoid_params['stride_window'] = stride_window
    bsoid_params['conf_threshold'] = 0.3
    bsoid = BSOID(**bsoid_params)
    bsoid.save()

    # bsoid = BSOID.load_config(**MODEL_LOAD_PARAMS)

    # bsoid.get_data(parallel=True)
    # bsoid.process_csvs()

    lookup_file = '/projects/kumar-lab/StrainSurveyPoses/StrainSurveyMetaList_2019-04-09.tsv'
    # bsoid.load_from_dataset(lookup_file, data_dir='/projects/kumar-lab/StrainSurveyPoses')

    # bsoid.features_from_points(parallel=True)

    bsoid.umap_reduce(reduced_dim=3, sample_size=int(1e7))
    bsoid.identify_clusters_from_umap(cluster_range=[0.1, 1.2, 12])

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
    main()
    # validate_and_train(**MODEL_LOAD_PARAMS)
    # results()
    # get_cluster_information()