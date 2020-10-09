import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
from BSOID.bsoid import BSOID

bsoid = BSOID.load_config('/home/dhruvlaad/data', 'dis')

feats = bsoid.load_features(collect=False)
feats = np.vstack(feats)

# split data according to displacement threshold
dis_threshold = 0.5
displacements = feats[:,8:15].mean(axis=1)

active_idx = np.where(displacements >= 0.5)[0]
inactive_idx = np.where(displacements < 0.5)[0]
active_feats = feats[active_idx]
inactive_feats = feats[inactive_idx]

logging.info(f'divided data into active ({round(active_feats.shape[0]/feats.shape[0], 2)} %) and in-active ({round(inactive_feats.shape[0]/feats.shape[0], 2)} %) based on displacement threshold of {dis_threshold}')
bsoid = BSOID('dis_split', '/home/dhruvlaad/data', 0.3, 30, None, 3, None)
bsoid.save()

import joblib
with open(bsoid.output_dir + '/' + bsoid.run_id + '_features.sav', 'wb') as f:
    joblib.dump(active_feats, f)