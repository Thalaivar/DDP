import os
import joblib
import subprocess

from bsoid_stability_test import bsoid_stabilitytest_predictions

def execute_all_my_runs(nruns=50):
    base_dir = "/fastscratch/laadd/my_stability"
    config_file = "../config/config.yaml"

    for i in range(nruns):
        job_command = f"python ./mystability_tests.py --config {config_file} --run-id {i} --base-dir {base_dir}"
        with open("./my_stability.sh", 'a') as f:
            f.write('\n')
            f.write(job_command) 
        
        launch_cmd = f"sbatch ./my_stability.sh"
        subprocess.call(launch_cmd, shell=True)

        with open("./my_stability.sh", 'r') as f:
            lines = f.readlines()
        lines = lines[:-1]
        lines[-1] = lines[-1][:-1]
        with open("./my_stability.sh", 'w') as f:
            f.writelines(lines)

# before running, check if ./bsoid_stability.sh has correct job parameters
def execute_all_bsoid_runs(nruns=50):
    base_dir = "/fastscratch/laadd/bsoid_stability"
    train_size = 500000
    config_file = "../config/config.yaml"

    for i in range(nruns):
        job_command = f"python ./bsoid_stability_test.py --config {config_file} --run-id {i} --base-dir {base_dir} --train-size {train_size}"
        with open("./bsoid_stability.sh", 'a') as f:
            f.write('\n')
            f.write(job_command) 
        
        launch_cmd = f"sbatch ./bsoid_stability.sh"
        subprocess.call(launch_cmd, shell=True)

        with open("./bsoid_stability.sh", 'r') as f:
            lines = f.readlines()
        lines = lines[:-1]
        lines[-1] = lines[-1][:-1]
        with open("./bsoid_stability.sh", 'w') as f:
            f.writelines(lines)
    
def cleanup_bsoid_runs():
    base_dir = "/fastscratch/laadd/bsoid_stability"
    test_size = 100000
    config_file = "../config/config.yaml"

    models = []
    for f in os.listdir(base_dir):
        if f.endswith(".model"):
            with open(os.path.join(base_dir, f), "rb") as ff:
                models.append(joblib.load(ff))
            os.remove(os.path.join(base_dir, f))
    
    bsoid_stabilitytest_predictions(models, config_file, test_size, base_dir)