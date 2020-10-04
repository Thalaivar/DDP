import logging

logging.basicConfig(level=logging.INFO)

from BSOID.bsoid import BSOID

bsoid_params = {'run_id': 'bsoid_feats', 
                'base_dir': 'D:/IIT/DDP/data',
                'conf_threshold': 0.3,
                'fps': 30,
                'stride_window': 3}

bsoid = BSOID(**bsoid_params)

bsoid.bsoid_features()