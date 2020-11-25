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

results = bsoid.identify_clusters_from_umap(cluster_range=[0.1, 1.0, 10])

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
labels = results[2].astype('int')
prop = [0 for _ in range(labels.max() + 1)]
for idx in labels:
    prop[idx] += 1
prop = np.array(prop)
prop = prop/prop.sum()
sns.barplot(x=np.arange(labels.max() + 1), y=prop)
plt.show()

# bsoid.validate_classifier()
# bsoid.train_classifier()

# bsoid.label_frames()