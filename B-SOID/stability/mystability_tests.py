import sys
import shutil
import pysftp
from getpass import getpass
from tqdm import tqdm

sys.path.insert(0, "/Users/dhruvlaad/IIT/DDP/DDP/B-SOID/")

from BSOID.bsoid import *

def mystability_train_model(config_file, run_id, base_dir):
    feats = BSOID(config_file).load_features(collect=False)

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["run_id"] = f"run-{run_id}"
    config["base_dir"] = base_dir
    bsoid = BSOID(config)

    with open(os.path.join(bsoid.output_dir, f"{bsoid.run_id}_features.sav"), "wb") as f:
        joblib.dump(feats, f)
    del feats

    os.environ["PYTHONPATH"] = "/home/laadd/DDP/B-SOID/:" + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = "/home/laadd/DDP/B-SOID/BSOID:" + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = "/home/laadd/DDP/B-SOID/stability:" + os.environ.get("PYTHONPATH", "")

    bsoid.cluster_strainwise()
    bsoid.pool()
    model = bsoid.train()
    
    with open(os.path.join(base_dir, f"{bsoid.run_id}.model"), "rb") as f:
        joblib.dump(model, f)
    
    shutil.rmtree(bsoid.base_dir, ignore_errors=True)

def download_datasets(base_dir):
    password = getpass("JAX ssh login password: ")
    with pysftp.Connection('login.sumner.jax.org', username='laadd', password=password) as sftp:
        files = sftp.listdir(base_dir)
        for i in tqdm(range(len(files))):
            f = files[i]
            if f.endswith("dataset.sav"):
                sftp.get(os.path.join(base_dir, f))

def mystabilitytest_predictions(models, config_file, test_size, base_dir):
    feats = []
    for _, data in BSOID(config_file).load_features(collect=False).items():
        feats.extend(data)
    feats = np.vstack(feats)

    feats = feats[np.random.choice(feats.shape[0], test_size, replace=False)]
    labels = [clf.predict(feats) for clf in models]
    with open(os.path.join(base_dir, "my_runs.labels"), "wb") as f:
        joblib.dump(labels, f)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser("mystability_tests.py")
    parser.add_argument("--config", type=str)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--base-dir", type=str)
    args = parser.parse_args()

    config_file = args.config
    run_id = args.run_id
    base_dir = args.base_dir

    mystability_train_model(config_file, run_id, base_dir)