import sys
sys.path.insert(0, "/home/laadd/DDP/B-SOID/")

from BSOID.bsoid import *

def mystability_train_model(config_file, run_id, base_dir, train_size)
    feats = BSOID(config_file).load_features(collect=False)

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["run_id"] = f"run-{run_id}"
    config["base_dir"] = base_dir
    bsoid = BSOID(config)

    with open(os.path.join(bsoid.output_dir, f"{bsoid.run_id}_features.sav"), "wb") as f:
        joblib.dump(fdata, f)
    del feats

    bsoid.cluster_strainwise()
    