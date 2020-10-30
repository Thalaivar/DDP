import joblib
from tqdm import tqdm
from BSOID.bsoid import BSOID

def generate_umap_embeddings(n: int, output_dir: str):
    base_dir = '/home/dhruvlaad/data'
    bsoid = BSOID.load_config(base_dir, run_id='dis')

    if not hasattr(bsoid, 'active_split'):
        bsoid.active_split = False
        
    for i in tqdm(range(n)):
        results = bsoid.umap_reduce_all(reduced_dim=3, sample_size=int(4e5))
        with open(output_dir + f'/umap_{i}.pkl', 'wb') as f:
            joblib.dump(results, f)