import matplotlib as mpl
mpl.use('tKAgg')

import joblib
import logging
from BSOID.bsoid import BSOID

logging.basicConfig(level=logging.INFO)

bsoid_params = {'run_id': 'temporal_feats',
                'base_dir': '/Users/dhruvlaad/IIT/DDP/data',
                'conf_threshold': 0.3,
                'fps': 30,
                'temporal_window': 16,
                'stride_window': 3,
                'temporal_dims': 7}
bsoid = BSOID(**bsoid_params)

# bsoid.process_csvs()
# bsoid.features_from_points()

# original workflow is to use only subset of data
# bsoid.max_samples_for_umap()
# bsoid.umap_reduce(reduced_dim=10, sample_size=int(6e5))

# bsoid.identify_clusters_from_umap(min_cluster_prop=0.28)

# bsoid.validate_classifier()
bsoid.train_classifier()

# bsoid.label_frames()