import time
import subprocess
from BSOID.bsoid import *
from BSOID.similarity import *

def coassociation_matrix(labels):
    mat = np.repeat(labels.reshape(1,-1), len(labels), axis=0)
    sim_mat = np.equal(mat, mat.T).astype(int)
    return sim_mat

def estimate_similarity_between_clusters(templates, labels, points_per_class=600):
    classes = np.unique(labels)
    
    X, labels = find_templates(labels, templates, num_points=points_per_class * (classes.max() + 1))
    
    clusters = []
    for lab in np.unique(labels):
        clusters.append(X[np.where(labels == lab)[0]])
    
    def par_sim(i, j, clusters):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = density_separation_similarity(clusters[i], clusters[j], metric="euclidean")
            
        return [sim, i, j]
    
    nclasses = len(clusters)
    sim = np.array(Parallel(n_jobs=psutil.cpu_count(logical=False))(delayed(par_sim)(i, j, clusters) for i, j in combinations(range(nclasses), 2)))
    
    sim_mat = np.zeros((nclasses, nclasses))
    for i in range(sim.shape[0]):
        c1, c2 = sim[i, 1:].astype(int)
        sim_mat[c1, c2] = sim[i,0]
        sim_mat[c2, c1] = sim[i,0]
        
    return sim_mat

def evidence_accumulation_run(run_id, base_dir, config_file):
    bsoid = BSOID(config_file)
    templates, _ = bsoid.load_pooled_dataset()
    with open("/home/laadd/data/models/max_label.model", "rb") as f:
        model = joblib.load(f)
    clustering = model.predict(templates).flatten()

    sim_mat = estimate_similarity_between_clusters(templates, clustering, points_per_class=600)
    embedding = umap.UMAP(metric="precomputed", min_dist=0.0, n_neighbors=2).fit_transform(sim_mat)
    labels = cluster_with_hdbscan(embedding, verbose=False, min_samples=1, prediction_data=True, cluster_range=[3, 5, 5])[2]

    np.save(os.path.join(base_dir, f"run-{run_id}-labels.npy"), labels)

def evidence_accumulation_matrix_runs(nruns, base_dir, config_file, default_job_file):
    try: os.mkdir(base_dir)
    except FileExistsError: pass

    for i in range(nruns):
        job_command = f"python ./eac.py --config {config_file} --run-id {i} --base-dir {base_dir}"
        with open(default_job_file, 'a') as f:
            f.write('\n')
            f.write(job_command) 
        
        launch_cmd = f"sbatch {default_job_file}"
        subprocess.call(launch_cmd, shell=True)

        with open(default_job_file, 'r') as f:
            lines = f.readlines()
        lines = lines[:-1]
        lines[-1] = lines[-1][:-1]
        with open(default_job_file, 'w') as f:
            f.writelines(lines)

        time.sleep(60)

def collect_runs(base_dir):
    labels = [np.load(os.path.join(base_dir, f)) for f in os.listdir(base_dir) if f.endswith("labels.npy")]
    print(f"Found {len(labels)} runs")
    nclusters = labels[0].size
    eac_mat = np.zeros((nclusters, nclusters))
    for lab in labels:
        eac_mat += coassociation_matrix(lab)
    
    eac_mat /= len(labels)
    dissim_mat = np.abs(eac_mat.max() - eac_mat)

    np.save(os.path.join(base_dir, "eac_mat.npy", dissim_mat)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("eac.py")
    parser.add_argument("--config", type=str)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--base-dir", type=str)
    args = parser.parse_args()

    evidence_accumulation_run(args.run_id, args.base_dir, args.config)