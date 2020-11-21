import joblib
import logging
from BSOID.bsoid import BSOID

logging.basicConfig(level=logging.INFO)

# bsoid_params = {'run_id': 'temporal_new_geo',
#                 'base_dir': '/home/dhruvlaad/data',
#                 'conf_threshold': 0.3,
#                 'fps': 30,
#                 'temporal_window': 16,
#                 'stride_window': 3,
#                 'temporal_dims': 6}
# bsoid = BSOID(**bsoid_params)

bsoid = BSOID.load_config(base_dir='D:/IIT/DDP/data', run_id='dis')
# bsoid.get_data()
# bsoid.process_csvs()

# bsoid.features_from_points()

# original workflow is to use only subset of data
# bsoid.max_samples_for_umap()
# bsoid.umap_reduce(reduced_dim=10, sample_size=int(6e5))

bsoid.identify_clusters_from_umap(cluster_range=0.06)

# bsoid.validate_classifier()
# bsoid.train_classifier()

# bsoid.label_frames()