import joblib
import logging
from BSOID.bsoid import BSOID

logging.basicConfig(level=logging.INFO)

bsoid = BSOID('vinit_feats', 'D:/IIT/DDP/data_custom')
# bsoid.umap_reduce(reduced_dim=10, sample_size=int(7e5), shuffle=True)
bsoid.cluster_feats(min_cluster_prop=0.3, reduced_feats=False, scale_feats=True)
# bsoid.validate_classifier()