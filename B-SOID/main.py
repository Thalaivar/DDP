import joblib
import logging
from BSOID.bsoid import BSOID

logging.basicConfig(level=logging.INFO)

bsoid = BSOID('vinit_feats', '/home/dhruvlaad/data_custom')
bsoid.umap_reduce(reduced_dim=10, sample_size=int(7e5), shuffle=True)
bsoid.cluster_feats(reduced_feats=True)