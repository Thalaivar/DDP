import logging
from BSOID.bsoid import BSOID
from BSOID.clustering import *

logging.basicConfig(level=logging.DEBUG)

bsoid = BSOID.load_config('/home/dhruvlaad/data', 'temporal_feats')

feats, _ = bsoid.load_features()

partitions, assignments = preclustering(feats, n_parts=10, min_clusters=75, max_clusters=100)
clusters = clusters_from_assignments(partitions, assignments, n_rep=1000, alpha=0.5)

cure = bigCURE(desired_clusters=25, n_rep=1000, alpha=0.5)
cure.fit(clusters)