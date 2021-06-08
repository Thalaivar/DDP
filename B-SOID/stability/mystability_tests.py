import sys
import shutil
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
    bsoid.pool()
    model = bsoid.train()

    with open(os.path.join(base_dir, f"run-{run_id}.model"), "wb") as f:
        joblib.dump(model, f)
    
    shutil.rmtree(bsoid.base_dir, ignore_errors=True)

def bsoid_stabilitytest_predictions(models, config_file, test_size, base_dir):
    feats = []
    for _, data in BSOID(config_file).load_features(collect=False).items():
        feats.extend(data)
    feats = np.vstack(feats)

    feats = feats[np.random.choice(feats.shape[0], test_size, replace=False)]
    labels = [clf.predict(feats) for clf in models]
    with open(os.path.join(base_dir, "my_runs.labels"), "wb") as f:
        joblib.dump(labels, f)