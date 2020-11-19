MLP_PARAMS = {
    'hidden_layer_sizes': (100, 10),  # 100 units, 10 layers
    'activation': 'logistic',  # logistics appears to outperform tanh and relu
    'solver': 'adam',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,  # learning rate not too high
    'alpha': 0.0001,  # regularization default is better than higher values.
    'max_iter': 1000,
    'early_stopping': False,
    'verbose': 1  # set to 1 for tuning your feedforward neural network
}

UMAP_PARAMS = {
    'min_dist': 0.0,  # small value
    'random_state': 23,
}

HDBSCAN_PARAMS = {
    'min_samples': 10,
    'prediction_data': True,
}

BASE_DIR = '/home/dhruvlaad/data'
DIRS = {
    'output':  BASE_DIR + '/output',
    'test': BASE_DIR + '/test',
    'raw': BASE_DIR + '/raw',
    'csv': BASE_DIR + '/csv'
}

