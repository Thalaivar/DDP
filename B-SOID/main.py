import joblib
import logging
from BSOID.bsoid import BSOID

logging.basicConfig(level=logging.INFO)

bsoid = BSOID('vinit_feats', '/scratch/scratch2/dhruvlaad/bsoid')
bsoid.process_csvs(parallel=False)
bsoid.features_from_points(temporal_dims=7)
bsoid.cluster_feats(scale_feats=True)