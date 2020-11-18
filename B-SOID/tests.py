import umap
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

RUN_ID = 'split'
BASE_DIR = 'D:/IIT/DDP/data'
DIS_THRESH = 1.0
SAMPLE_SIZE = int(7e5)

import joblib
from joblib import delayed, Parallel
from sklearn.preprocessing import StandardScaler
from active_inactive_split import calc_dis_threshold

n_neighbors = [100, 300, 600, 800]

with open(f'{BASE_DIR}/output/{RUN_ID}_features.sav', 'rb') as f:
    active_feats, inactive_feats = joblib.load(f)

logging.info(f'active feats have {active_feats.shape[0]} samples and inactive feats have {inactive_feats.shape[0]} samples')
scaler = StandardScaler().fit(np.vstack((active_feats, inactive_feats)))

displacements = calc_dis_threshold(active_feats)
assert np.any(displacements < DIS_THRESH)
displacements = calc_dis_threshold(inactive_feats)
assert np.any(displacements >= DIS_THRESH)

del inactive_feats
del displacements

def embed(feats, scaler, n_neighbors):
    feats_sc = scaler.transform(feats)
    if SAMPLE_SIZE > 0 and SAMPLE_SIZE < feats_sc.shape[0]:
        idx = np.random.permutation(np.arange(feats_sc.shape[0]))[0:SAMPLE_SIZE]
        feats_train = feats_sc[idx,:]
        feats_usc = feats[idx, :]
    else:
        feats_train = feats_sc
        feats_usc = feats

    logging.info('running UMAP on {} samples with n_neighbors={}'.format(*feats_train.shape, n_neighbors))
    mapper = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.0).fit(feats_train)

    with open(f'/home/dhruvlaad/umap_test_nbrs_{n_neighbors}.sav', 'wb') as f:
        joblib.dump([feats_usc, mapper.embedding_], f)

Parallel(n_jobs=2)(delayed(embed)(active_feats, scaler, nbrs) for nbrs in n_neighbors)