import joblib
import logging
from BSOID.bsoid import BSOID

logging.basicConfig(level=logging.INFO)


model_load_params = None
# model_load_params = {'run_id': 'dis', 'base_dir': '/home/dhruvlaad/data'}

get_data = True
preprocess = True
extract_features = True
embed = True

if model_load_params is None:
    bsoid_params = {'run_id': 'dis',
                    'base_dir': '/home/dhruvlaad/data',
                    'fps': 30,
                    'stride_window': 3,
                    'conf_threshold': 0.3
        }
    bsoid = BSOID(**bsoid_params)
    bsoid.save()
else:
    bsoid = BSOID.load_config(**model_load_params)

if get_data:
    bsoid.get_data(parallel=True)
if preprocess:
    bsoid.process_csvs()
if extract_features:
    bsoid.features_from_points(parallel=True)
if embed:
    bsoid.best_reduced_dim()
    reduced_dim = int(input('Enter reduced dimensions for embedding: '))
    bsoid.umap_reduce(reduced_dim=reduced_dim, sample_size=int(7e5))

# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# labels = results[2].astype('int')
# prop = [0 for _ in range(labels.max() + 1)]
# for idx in labels:
#     prop[idx] += 1
# prop = np.array(prop)
# prop = prop/prop.sum()
# sns.barplot(x=np.arange(labels.max() + 1), y=prop)
# plt.show()

# bsoid.validate_classifier()
# bsoid.train_classifier()

# bsoid.label_frames()