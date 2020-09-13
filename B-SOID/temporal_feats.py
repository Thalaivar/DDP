# determine optimal number of temporal features to retain
import joblib
import logging
from BSOID.bsoid import BSOID
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

bsoid = BSOID('vinit_feats', '/Users/dhruvlaad/IIT/DDP/data_custom')
# bsoid.get_data()
# bsoid.process_csvs()
# _, temporal_feats = bsoid.features_from_points()

with open(bsoid.output_dir + '/' + bsoid.run_id + '_features.sav', 'rb') as f:
    feats, temporal_feats = joblib.load(f)

logging.info('running PCA with multiple dimensions to check variance retained.')
dims = []
var = []
for i in tqdm(range(1, 30)):
    dims.append(i)
    pca = PCA(n_components=i, random_state=0).fit(temporal_feats)
    train_pca = pca.transform(temporal_feats)
    var.append(pca.explained_variance_ratio_.sum())

plt.plot(dims, var)
plt.xlabel('Dimensions')
plt.ylabel('Variance Retained')
plt.title('Determine components retained by PCA')
plt.show()