import joblib
import logging
from BSOID.bsoid import BSOID

logging.basicConfig(level=logging.INFO)

bsoid = BSOID('vinit_feats', '/Users/dhruvlaad/IIT/DDP/data_custom')
bsoid.cluster_feats()