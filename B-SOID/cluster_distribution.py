import matplotlib as mpl
mpl.use('tKAgg')

import joblib
import logging
from BSOID.bsoid import BSOID

logging.basicConfig(level=logging.INFO)

bsoid = BSOID.load_config('/Users/dhruvlaad/IIT/DDP/data', 'temporal_feats')

# bsoid.identify_clusters_from_umap(min_cluster_prop=0.33)
assignments, soft_clusters, soft_assignments = bsoid.load_identified_clusters()

import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
sn.set_theme()

cluster_data = pd.Series(data=soft_assignments, name='Clusters')
sn.displot(cluster_data)
plt.show()