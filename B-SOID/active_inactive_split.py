import joblib
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

from BSOID.bsoid import *
from sklearn.preprocessing import StandardScaler
from BSOID.utils import cluster_with_hdbscan

RUN_ID = 'split'
BASE_DIR = '/home/laadd/data'

CLUSTER_PARAMS = {
    'active': [0.1, 1.0],
    'inactive': [0.1, 1.0]
}

def calc_dis_threshold(feats):
    head_dis = feats[:,7].reshape(-1,1)
    tail_dis = feats[:,12:15].mean(axis=1).reshape(-1,1)
    displacements = np.hstack((head_dis, tail_dis)).mean(axis=1).reshape(-1,1)
    return displacements

def split_data(dis_threshold: float):
    bsoid = BSOID.load_config(BASE_DIR, 'dis')

    feats, feats_sc = bsoid.load_features()

    # split data according to displacement threshold
    displacements = calc_dis_threshold(feats)

    active_idx = np.where(displacements >= dis_threshold)[0]
    inactive_idx = np.where(displacements < dis_threshold)[0]

    # create bsoid model for later use
    print(f'divided data into active ({round(active_idx.shape[0]/feats.shape[0], 2) * 100}%) and in-active ({round(inactive_idx.shape[0]/feats.shape[0], 2) * 100}%) based on displacement threshold of {dis_threshold}')
    bsoid = BSOID(RUN_ID, BASE_DIR, fps=30, conf_threshold=0.3)
    bsoid.save()

    active_feats = [feats[active_idx], feats_sc[active_idx]]
    inactive_feats = [feats[inactive_idx], feats_sc[inactive_idx]]

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_features.sav', 'wb') as f:
        joblib.dump([active_feats, inactive_feats], f)

def embed_split_data(reduced_dim: int, sample_size: int, dis_threshold=None):
    bsoid = BSOID.load_config(BASE_DIR, RUN_ID)

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_features.sav', 'rb') as f:
        active_feats, inactive_feats = joblib.load(f)

    logging.info(f'active feats have {active_feats[0].shape[0]} samples and inactive feats have {inactive_feats[0].shape[0]} samples')
    
    active_feats, active_feats_sc = active_feats
    inactive_feats, inactive_feats_sc = inactive_feats

    if dis_threshold is not None:
            displacements = calc_dis_threshold(active_feats)
            assert not np.any(displacements < dis_threshold)

            displacements = calc_dis_threshold(inactive_feats)
            assert not np.any(displacements >= dis_threshold)

    comb_feats = [active_feats, inactive_feats]
    comb_feats_sc = [active_feats_sc, inactive_feats_sc]

    umap_results = []
    for i in range(2):
        if sample_size > 0 and sample_size < comb_feats[i].shape[0]:
            idx = np.random.permutation(np.arange(comb_feats[i].shape[0]))[0:sample_size]
            feats_train = comb_feats_sc[i][idx,:]
            feats_usc = comb_feats[i][idx, :]
        else:
            feats_train = comb_feats_sc[i]
            feats_usc = comb_feats[i]

        logging.info('running UMAP on {} samples from {}D to {}D'.format(*feats_train.shape, reduced_dim))
        mapper = umap.UMAP(n_components=reduced_dim, **UMAP_PARAMS).fit(feats_train)
        umap_results.append([feats_usc, feats_train, mapper.embedding_])
        
    with open(bsoid.output_dir + '/' + bsoid.run_id + '_umap.sav', 'wb') as f:
        joblib.dump(umap_results, f)

def cluster_split_data():
    bsoid = BSOID.load_config(BASE_DIR, RUN_ID)

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_umap.sav', 'rb') as f:
        umap_results = joblib.load(f)

    comb_assignments, comb_soft_clusters, comb_soft_assignments, comb_clf = [], [], [], []
    for i, results in enumerate(umap_results):
        _, _, umap_embeddings = results
        
        # cluster_range = [float(x) for x in input('Enter cluster range: ').split()]
        if i == 0:
            cluster_range = CLUSTER_PARAMS['active']
        elif i == 1:
            cluster_range = CLUSTER_PARAMS['inactive']
            
        assignments, soft_clusters, soft_assignments, best_clf = cluster_with_hdbscan(umap_embeddings, cluster_range, HDBSCAN_PARAMS)
        comb_assignments.append(assignments)
        comb_soft_clusters.append(soft_clusters)
        comb_soft_assignments.append(soft_assignments)
        comb_clf.append(best_clf)

        logging.info('identified {} clusters from {} samples in {}D'.format(len(np.unique(soft_assignments)), *umap_embeddings.shape))

    n_1, n_2 = comb_assignments[0].size, comb_assignments[1].size
    n_cluster_1, n_cluster_2 = comb_soft_clusters[0].shape[1], comb_soft_clusters[1].shape[1]
    assignments = np.zeros((n_1+n_2,))
    assignments[:n_1] = comb_assignments[0]
    for i in range(n_2):
        if comb_assignments[1][i] >= 0:
            comb_assignments[1][i] += comb_assignments[0].max() + 1
    assignments[n_1:] = comb_assignments[1]

    soft_clusters = -1 * np.ones((n_1 + n_2, n_cluster_1 + n_cluster_2))
    soft_clusters[:n_1,:n_cluster_1] = comb_soft_clusters[0]
    soft_clusters[n_1:, n_cluster_1:] = comb_soft_clusters[1]

    soft_assignments = np.zeros((n_1 + n_2,))
    soft_assignments[:n_1] = comb_soft_assignments[0]
    soft_assignments[n_1:] = comb_soft_assignments[1] + comb_soft_assignments[0].max() + 1

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_clusters.sav', 'wb') as f:
        joblib.dump([assignments, soft_clusters, soft_assignments, comb_clf], f)

    return assignments, soft_clusters, soft_assignments

def train_classifier():
    bsoid = BSOID.load_config(BASE_DIR, RUN_ID)
    _, _, soft_assignments, _ = bsoid.load_identified_clusters()

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_umap.sav', 'rb') as f:
        umap_results = joblib.load(f)

    feats_sc = np.vstack([data[1] for data in umap_results])

    # shuffle dataset
    shuffled_idx = np.random.permutation(np.arange(feats_sc.shape[0]))
    feats_sc, soft_assignments = feats_sc[shuffled_idx], soft_assignments[shuffled_idx]

    logging.info('training neural network on {} scaled samples in {}D'.format(*feats_sc.shape))
    clf = MLPClassifier(**MLP_PARAMS).fit(feats_sc, soft_assignments)

    with open(bsoid.output_dir + '/' + bsoid.run_id + '_classifiers.sav', 'wb') as f:
        joblib.dump(clf, f)

    
def validate_classifier():
    bsoid = BSOID.load_config(BASE_DIR, RUN_ID)
    _, _, soft_assignments, _ = bsoid.load_identified_clusters()
    
    with open(bsoid.output_dir + '/' + bsoid.run_id + '_umap.sav', 'rb') as f:
        umap_results = joblib.load(f)

    feats_sc = np.vstack([data[1] for data in umap_results])
    shuffled_idx = np.random.permutation(np.arange(feats_sc.shape[0]))
    feats_sc, soft_assignments = feats_sc[shuffled_idx], soft_assignments[shuffled_idx]
    
    logging.info('validating classifier on {} features'.format(*feats_sc.shape))

    feats_train, feats_test, labels_train, labels_test = train_test_split(feats_sc, soft_assignments)
    clf = MLPClassifier(**MLP_PARAMS).fit(feats_train, labels_train)
    sc_scores = cross_val_score(clf, feats_test, labels_test, cv=5, n_jobs=-1)
    sc_cf = create_confusion_matrix(feats_test, labels_test, clf)
    logging.info('classifier accuracy: {} +- {}'.format(sc_scores.mean(), sc_scores.std())) 
        
    with open(bsoid.output_dir + '/' + bsoid.run_id + '_validation.sav', 'wb') as f:
        joblib.dump([sc_scores, sc_cf], f)

if __name__ == "__main__":
    # embed_split_data(reduced_dim=3, sample_size=-1, dis_threshold=2.0)

    cluster_split_data()
    validate_classifier()
    train_classifier()