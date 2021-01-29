import os
import joblib
import ftplib
import h5py
import numpy as np
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
from joblib import Parallel, delayed
from BSOID.data import bsoid_format, get_pose_data_dir
from BSOID.preprocessing import likelihood_filter
from BSOID.prediction import *
from sklearn.neural_network import MLPClassifier

FEATS_TYPE = 'dis'
STRAINS = ["LL6-B2B", "LL5-B2B", "LL4-B2B", "LL3-B2B", "LL2-B2B", "LL1-B2B"]
DATASETS = ["strain-survey-batch-2019-05-29-e/", "strain-survey-batch-2019-05-29-d/", 
            "strain-survey-batch-2019-05-29-c/", "strain-survey-batch-2019-05-29-b/",
            "strain-survey-batch-2019-05-29-a/"]

# BASE_DIR = '/home/laadd/data'
BASE_DIR = 'D:/IIT/DDP/data'
RAW_DIR = BASE_DIR + '/raw'
FPS = 30

BEHAVIOUR_LABELS = {
    'Groom': [0, 1, 2, 3, 5, 9],
    'Run': [6],
    'Walk': [8, 18],
    'CW-Turn': [7],
    'CCW-Turn': [10, 11],
    'Point': [12, 19],
    'Rear': [13, 15, 17],
    'N/A': [4, 14, 16]
}

MIN_BOUT_LENS = {
    'Groom': 3000,
    'Run': 250,
    'Walk': 200,
    'CW-Turn': 600,
    'CCW-Turn': 600,
    'Point': 200,
    'Rear': 250,
    'N/A': 200
}

for lab, bout_len in MIN_BOUT_LENS.items():
    MIN_BOUT_LENS[lab] = bout_len * FPS / 1000

try:
    os.makedirs(RAW_DIR)
except FileExistsError:
    pass

FPS = 30
STRIDE_WINDOW = 3

def get_mouse_raw_data(metadata: dict, pose_dir=None):
    pose_dir = RAW_DIR if pose_dir is None else pose_dir

    _, _, movie_name = metadata['NetworkFilename'].split('/')
    filename = f'{pose_dir}/{movie_name[0:-4]}_pose_est_v2.h5'
    f = h5py.File(filename, 'r')
    data = list(f.keys())[0]
    keys = list(f[data].keys())
    conf, pos = np.array(f[data][keys[0]]), np.array(f[data][keys[1]])
    f.close()

    # trim start and end
    data = _bsoid_format(conf, pos)
    fdata, perc_filt = likelihood_filter(data, fps=FPS, end_trim=2, clip_window=0, conf_threshold=0.3)

    strain, mouse_id = metadata['Strain'], metadata['MouseID']
    if perc_filt > 10:
        logging.warning(f'mouse:{strain}/{mouse_id}: % data filtered from raw data is too high ({perc_filt} %)')
    
    data_shape = fdata['x'].shape
    logging.info(f'preprocessed raw data of shape: {data_shape} for mouse:{strain}/{mouse_id}')

    return fdata

def extract_features(metadata: dict, pose_dir=None, features_type='dis'):
    data = get_mouse_raw_data(metadata, pose_dir)
    if features_type == 'dis':
        from BSOID.features.displacement_feats import extract_feats, window_extracted_feats
    
    feats = frameshift_features(data, STRIDE_WINDOW, FPS, extract_feats, window_extracted_feats)
    return feats

def get_behaviour_labels(metadata: dict, clf: MLPClassifier, pose_dir=None, return_feats=False):
    feats = extract_features(metadata, pose_dir)
    labels = frameshift_predict(feats, clf, STRIDE_WINDOW)
    if return_feats:
        return labels, feats
    else:
        return labels

def idx_2_behaviour(idx):
    for behaviour, idxs in BEHAVIOUR_LABELS.items():
        if idx in idxs:
            return behaviour
    
    return None

"""
    Helper functions for carrying out analysis per mouse:
        - transition_matrix_from_assay : calculates the transition matrix for a given assay
        - get_behaviour_info_from_assay : extracts statistics of all behaviours from given assay
"""
def get_stats_for_all_labels(labels: np.ndarray):
    stats = {}
    for key in BEHAVIOUR_LABELS.keys():
        stats[key] = {'TD': 0, 'ABL': [], 'NB': 0}
    
    i = 0
    while i < len(labels) - 1:
        curr_label = labels[i]
        curr_bout_len, j = 1, i + 1
        curr_behaviour = idx_2_behaviour(curr_label)

        while j < len(labels) - 1 and curr_label in BEHAVIOUR_LABELS[curr_behaviour]:
            curr_label = labels[j]
            curr_bout_len += 1
            j += 1
        
        if curr_bout_len >= MIN_BOUT_LENS[curr_behaviour]:
            stats[curr_behaviour]['ABL'].append(curr_bout_len)
            stats[curr_behaviour]['NB'] += 1
        
        i = j
    
    for lab, stat in stats.items():
        stats[lab]['TD'] = sum(stat['ABL']) / FPS
        stats[lab]['ABL'] = sum(stat['ABL'])/len(stat['ABL']) if len(stat['ABL']) > 0 else 0

        stats[lab]['ABL'] /= FPS

    return stats    

def transition_matrix_from_assay(labels, max_label=None):
    n_lab = labels.max() + 1 if max_label is None else max_label + 1
    tmat = np.zeros((n_lab, n_lab))
    curr_lab = labels[0]
    for i in range(1, labels.size):
        lab = labels[i]
        tmat[curr_lab, lab] += 1
        curr_lab = lab
    
    for i in range(n_lab):
        if tmat[i].sum() > 0:
            tmat[i] = tmat[i] / tmat[i].sum()


    return tmat

def behaviour_proportion(labels, n_lab=None):
    n_lab = labels.max() + 1 if n_lab is None else n_lab + 1
    
    prop = [0 for _ in range(n_lab)]
    for i in range(labels.size):
        prop[labels[i]] += 1
    
    prop = np.array(prop)
    prop = prop/prop.sum()

    return prop

"""
    modules to run different analyses for all mice
"""
def extract_labels_for_all_mice(data_lookup_file: str, clf_file: str, data_dir=None):
    with open(clf_file, 'rb') as f:
        clf = joblib.load(f)

    if data_lookup_file.endswith('.tsv'):
        data = pd.read_csv(data_lookup_file, sep='\t')    
    else:
        data = pd.read_csv(data_lookup_file)
    N = data.shape[0]

    def extract_(metadata, clf, data_dir):
        try:
            if data_dir is not None:
                pose_dir, _ = get_pose_data_dir(data_dir, metadata['NetworkFilename'])
            else:
                pose_dir = None
            labels_ = get_behaviour_labels(metadata, clf, pose_dir)
            return [metadata, labels_]
        except:
            return None

    labels = Parallel(n_jobs=-1)(delayed(extract_)(data.iloc[i], clf, data_dir) for i in range(N))

    all_strain_labels = {
        'Strain': [],
        'Sex': [],
        'MouseID': [],
        'NetworkFilename': [],
        'Labels': []
    }

    for l in labels:
        if l is not None:
            metadata, lab = l
            all_strain_labels['Strain'].append(metadata['Strain'])
            all_strain_labels['Sex'].append(metadata['Sex'])
            all_strain_labels['MouseID'].append(metadata['MouseID'])
            all_strain_labels['NetworkFilename'].append(metadata['NetworkFilename'])
            all_strain_labels['Labels'].append(lab)

    strains = all_strain_labels['Strain']
    print(f'Extracted labels for {len(strains)}/{N} mice')
    return all_strain_labels

def all_behaviour_info_for_all_strains(label_info_file: str, max_label: int):
    with open(label_info_file, 'rb') as f:
        label_info = joblib.load(f)
    N = len(label_info['Strain'])

    info = {}
    for key in BEHAVIOUR_LABELS.keys():
        info[key] = {
            'Strain': [], 
            'Sex': [], 
            'Total Duration':  [], 
            'Average Bout Length': [], 
            'No. of Bouts': []
        }

    def parallel_extract(labels, strain, sex):
        stats = get_stats_for_all_labels(labels)
        stats['Strain'] = strain
        stats['Sex'] = sex

        return stats
    
    import pathos.multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    stats = [pool.apply_async(parallel_extract, args=(label_info['Labels'][i], label_info['Strain'][i], label_info['Sex'][i])) for i in range(N)]
    pool.close()
    pool.join()
    
    for stat in stats:
        stat = stat.get()
        for behaviour in BEHAVIOUR_LABELS.keys():
            info[behaviour]['Strain'].append(stat['Strain'])
            info[behaviour]['Sex'].append(stat['Sex'])
            info[behaviour]['Total Duration'].append(stat[behaviour]['TD'])
            info[behaviour]['Average Bout Length'].append(stat[behaviour]['ABL'])
            info[behaviour]['No. of Bouts'].append(stat[behaviour]['NB'])

    return info

def calculate_behaviour_usage(label_info_file: str, max_label=None):
    with open(label_info_file, 'rb') as f:
        label_info = joblib.load(f)
    N = len(label_info['Strain'])

    prop = []
    for i in tqdm(range(N)):
        prop.append(behaviour_proportion(label_info['Labels'][i], max_label))
    prop = np.vstack(prop)
    prop = prop.sum(axis=0)/prop.shape[0]
    return prop

def behaviour_usage_across_strains(stats_file: str, min_threshold: float):
    with open(stats_file, 'rb') as f:
        info = joblib.load(f)
    strains = info['Groom']['Strain']
    N = len(strains)

    behaviour_usage = {
        'Behaviour': [],
        'Strain': [],
        'Usage': []
    }

    for i in range(N):
        total_duration = sum([info[behaviour]['Total Duration'][i] for behaviour in BEHAVIOUR_LABELS.keys()])
        
        for behaviour in BEHAVIOUR_LABELS.keys():
            behaviour_usage['Behaviour'].append(behaviour)
            behaviour_usage['Strain'].append(strains[i])
            
            usage = info[behaviour]['Total Duration'][i]/total_duration
            usage = usage if usage > min_threshold else min_threshold
            behaviour_usage['Usage'].append(usage)
    
    return pd.DataFrame.from_dict(behaviour_usage)

def calculate_transition_matrices_for_all_strains(label_info_file: str, max_label=None):
    with open(label_info_file, 'rb') as f:
        label_info = joblib.load(f)
    N = len(label_info['Strain'])

    tmat_data = {
        'Strain': [],
        'Sex': [],
        'MouseID': [],
        'NetworkFilename': [],
        'Transition Matrix': []
    }

    for key, val in label_info.items():
        if key in tmat_data.keys():
            tmat_data[key] = val

    for i in tqdm(range(N)):
        tmat_data['Transition Matrix'].append(transition_matrix_from_assay(label_info['Labels'][i], max_label))
    
    return tmat_data

def get_behaviour_trajectories(data_lookup_file, clf_file, n, n_trajs, data_dir, min_bout_len):
    with open(clf_file, 'rb') as f:
        clf = joblib.load(f)
    
    if data_lookup_file.endswith('.tsv'):
        data = pd.read_csv(data_lookup_file, sep='\t')
    else:
        data = pd.read_csv(data_lookup_file)
    N = data.shape[0]

    trajs = {}
    for key in BEHAVIOUR_LABELS.keys():
        trajs[key] = []

    idx = np.random.randint(0, N, n)
    for i in range(idx.shape[0]):
        metadata = data.iloc[idx[i]]
        if data_dir is not None:
            pose_dir, _ = get_pose_data_dir(data_dir, metadata['NetworkFilename'])
        else:
            pose_dir = None
        labels, feats = get_behaviour_labels(metadata, clf, pose_dir, return_feats=True)
        feats = feats[0]

        j = 0
        while j < len(labels):
            curr_label = labels[j]
            curr_bout_len, jj = 1, j + 1
            curr_behaviour = idx_2_behaviour(curr_label)
            while jj < len(labels) - 1 and curr_label in BEHAVIOUR_LABELS[curr_behaviour]:
                curr_label = labels[jj]
                curr_bout_len += 1
                jj += 1
            
            if curr_bout_len >= MIN_BOUT_LENS[curr_behaviour]:
                trajs[curr_behaviour].append(feats[j:jj])

            j = jj
    
    for lab in trajs.keys():
        trajs[lab].sort(reverse=True, key=lambda x: x.shape[0])
        trajs[lab] = trajs[lab][:n_trajs]

    return trajs


def calculate_autocorr()
    
if __name__ == "__main__":
    # lookup_file = '/projects/kumar-lab/StrainSurveyPoses/StrainSurveyMetaList_2019-04-09.tsv'
    lookup_file = 'bsoid_strain_data.csv'
    clf_file = f'{BASE_DIR}/output/dis_classifiers.sav'

    # info = extract_labels_for_all_mice(lookup_file, clf_file, data_dir='/projects/kumar-lab/StrainSurveyPoses')
    # info = extract_labels_for_all_mice(lookup_file, clf_file)
    # with open(f'{BASE_DIR}/analysis/label_info.pkl', 'wb') as f:
    #     joblib.dump(info, f)

    # label_info_file = f'{BASE_DIR}/analysis/label_info.pkl'
    # stats_file = f'{BASE_DIR}/analysis/stats.pkl'
    # stats = all_behaviour_info_for_all_strains(label_info_file, max_label=19)
    # with open(stats_file, 'wb') as f:
    #     joblib.dump(stats, f)

    # usage_data = behaviour_usage_across_strains(stats_file, min_threshold=0.01)
    trajs = get_behaviour_trajectories(lookup_file, clf_file, 20, 20, data_dir=None, min_bout_len=500)