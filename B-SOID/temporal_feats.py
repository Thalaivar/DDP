import logging
from BSOID.bsoid import BSOID
logging.basicConfig(level=logging.DEBUG)

bsoid = BSOID('vinit_feats', '/Users/dhruvlaad/IIT/DDP/data_custom')
bsoid.process_csvs()
bsoid.features_from_points()