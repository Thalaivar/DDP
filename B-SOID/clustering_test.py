import logging
import numpy as np
from BSOID.clustering import clusters_from_assignments
import joblib

logging.basicConfig(level=logging.DEBUG)

# with open('D:/IIT/DDP/data/output/all_umap_10D.sav', 'rb') as f:
#     _, _, umap_embeddings = joblib.load(f)

# print(umap_embeddings.shape)

# cure = bigCURE(25, 10, 100, 1000, 0.5)

# parts, assignments = cure.preclustering(umap_embeddings.astype(np.float64))
# with open('preclusters.sav', 'wb') as f:
#     joblib.dump([parts, assignments], f)

with open('preclusters.sav', 'rb') as f:
    parts, assignments = joblib.load(f)
clusters = clusters_from_assignments(parts, assignments, n_rep=1000, alpha=0.5)
with open('preclusters_1.sav', 'wb') as f:
    joblib.dump(clusters, f)
