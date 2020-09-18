import logging
from BSOID.clustering import CURE
import joblib

logging.basicConfig(level=logging.DEBUG)

with open('../../data/output/all_10D/all_umap_10D.sav', 'rb') as f:
    _, _, umap_embeddings = joblib.load(f)

print(umap_embeddings.shape)

cure = CURE(25, 100, 100, n_rep=1000, alpha=0.5)

clusters = cure.preclustering(umap_embeddings)
with open('preclusters.sav', 'wb') as f:
    joblib.dump(clusters, f)
