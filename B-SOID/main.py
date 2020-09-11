import joblib
import logging
from bsoid import BSOID
from preprocessing import normalize_feats

logging.basicConfig(level=logging.DEBUG)

bsoid = BSOID('test', '/Users/dhruvlaad/IIT/DDP/data_custom')
bsoid.cluster_feats(desired_clusters=25, n_parts=126)