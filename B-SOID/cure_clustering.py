import logging
import joblib
from BSOID.clustering import bigCURE

logging.basicConfig(level=logging.DEBUG)

with open('../../data/output/temporal_feats_clusters_all.sav', 'rb') as f:
    clusters = joblib.load(f)

cure = bigCURE(25, 1000, 0.5)
cure.init_w_clusters(clusters)