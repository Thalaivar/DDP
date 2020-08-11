# data storage
# BASE_PATH = '/scratch/scratch2/dhruvlaad/bsoid/'
BASE_PATH = '/Users/dhruvlaad/IIT/DDP/data/'
RAW_DATA_DIR = 'raw/'
CSV_DATA_DIR = 'test/'
OUTPUT_PATH = BASE_PATH + 'output/'

# run
FPS = 30
RETAIN_WINDOW = 30*60
MODEL_NAME = '8animals'

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