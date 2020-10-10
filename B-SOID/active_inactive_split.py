import logging
import joblib
import numpy as np
logging.basicConfig(level=logging.INFO)
from BSOID.bsoid import BSOID

bsoid = BSOID.load_config('/home/dhruvlaad/data', 'dis')

with open(bsoid.output_dir + '/' + bsoid.run_id + '_features.sav', 'rb') as f:
    feats = joblib.load(f)
feats = np.vstack(feats)

# split data according to displacement threshold
dis_threshold = 1.0
displacements = feats[:,7:15].mean(axis=1)

active_idx = np.where(displacements >= dis_threshold)[0]
inactive_idx = np.where(displacements < dis_threshold)[0]
active_feats = feats[active_idx]
inactive_feats = feats[inactive_idx]

logging.info(f'divided data into active ({round(active_feats.shape[0]/feats.shape[0], 2)}%) and in-active ({round(inactive_feats.shape[0]/feats.shape[0], 2)}%) based on displacement threshold of {dis_threshold}')
bsoid = BSOID('dis_active', '/home/dhruvlaad/data', 30, None, 3, None, True)
bsoid.save()

import joblib
with open(bsoid.output_dir + '/' + bsoid.run_id + '_features.sav', 'wb') as f:
    joblib.dump([active_feats, inactive_feats], f)