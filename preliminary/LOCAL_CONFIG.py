# data storage
BASE_PATH = '/Users/dhruvlaad/IIT/DDP/data/'
# BASE_PATH = '/home/dhruvlaad/data/'
RAW_DATA_DIR = 'raw/'
CSV_DATA_DIR = 'preproc/'
OUTPUT_PATH = BASE_PATH + 'output/'

# run
FPS = 30
RETAIN_WINDOW = 30*60
MODEL_NAME = 'incremental_plaw'
HLDOUT = 0.2
CV_IT = 10

# embedding
UMAP_PARAMS = {
    'n_components': 3,
    'min_dist': 0.0,  # small value
    'random_state': 23,
}

# clustering
HDBSCAN_PARAMS = {
    'min_samples': 10  # small value
}

# classifier
MLP_PARAMS = {
    'hidden_layer_sizes': (100, 10),  # 100 units, 10 layers
    'activation': 'logistic',  # logistics appears to outperform tanh and relu
    'solver': 'adam',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,  # learning rate not too high
    'alpha': 0.0001,  # regularization default is better than higher values.
    'max_iter': 1000,
    'early_stopping': False,
    'verbose': 0  # set to 1 for tuning your feedforward neural network
}

# testing
TEST_DIR = BASE_PATH + 'test/'