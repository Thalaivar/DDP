import sys
import shutil
import pysftp
from getpass import getpass
from tqdm import tqdm

sys.path.insert(0, "/home/laadd/DDP/B-SOID/")

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
    templates, clustering = bsoid.pool()
    
    with open(os.path.join(base_dir, f"{bsoid.run_id}.dataset"), "wb") as f:
        joblib.dump([templates, clustering], f)
    
    shutil.rmtree(bsoid.base_dir, ignore_errors=True)


def mystabilitytest_predictions(config_file, test_size, base_dir):
    bsoid = BSOID(config_file)

    feats = []
    for _, data in bsoid.load_features(collect=False).items():
        feats.extend(data)
    feats = np.vstack(feats)

    feats = feats[np.random.choice(feats.shape[0], test_size, replace=False)]

    labels = []
    dataset_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith("dataset")]
    for f in dataset_files:
        with open(f, "rb") as ff: 
            templates, clustering = joblib.load(ff)
        
        model = CatBoostClassifier(**bsoid.clf_params)
        model.fit(templates, clustering["soft_labels"])
        labels.append(model.predict(feats))
        
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