from pickle import STRING
from typing import MutableMapping
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import diff
from BSOID.bsoid import BSOID
import os
import joblib
import ftplib
import h5py
import numpy as np
import pandas as pd
import random

import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
from joblib import Parallel, delayed
from BSOID.data import bsoid_format, get_pose_data_dir
from BSOID.preprocessing import likelihood_filter
from prediction import *
from BSOID.features import extract_dis_feats as extract_feats
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    MIN_BOUT_LENS[lab] = bout_len * FPS // 1000

try:
    os.makedirs(RAW_DIR)
except FileExistsError:
    pass

FPS = 30
STRIDE_WINDOW = 3

def idx2group_map():
    i, idx2grp = 0, {}
    for lab, idxs in BEHAVIOUR_LABELS.items():
        for idx in idxs:
            idx2grp[idx] = i
        i += 1
    return idx2grp

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
    data = bsoid_format(conf, pos)
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

def idx_2_behaviour(idx, diff=False):
    for behaviour, idxs in BEHAVIOUR_LABELS.items():
        if not diff:
            if idx in idxs:
                return behaviour
        else:
            for i, j in enumerate(idxs):
                if idx == j:
                    return f'{behaviour} #{i}'
    
    return None

def get_max_label():
    max_label = -1
    for _, idxs in BEHAVIOUR_LABELS.items():
        max_label = max(max_label, max(idxs))
    
    return max_label

"""
    Helper functions for carrying out analysis per mouse:
        - transition_matrix_from_assay : calculates the transition matrix for a given assay
        - get_behaviour_info_from_assay : extracts statistics of all behaviours from given assay
"""
def get_stats_for_all_labels(labels: np.ndarray):
    stats = {}
    for i in range(get_max_label() + 1):
        stats[idx_2_behaviour(i, diff=True)] = {'TD': None, 'ABL': [], 'NB': 0}
    
    i = 0
    while i < len(labels) - 1:
        curr_label = labels[i]
        curr_bout_len, j = 1, i + 1
        curr_behaviour = idx_2_behaviour(curr_label, diff=True)

        while (j < len(labels) - 1) and (labels[j] == curr_label):
            curr_label = labels[j]
            curr_bout_len += 1
            j += 1
            
        if curr_bout_len >= MIN_BOUT_LENS[idx_2_behaviour(curr_label)]:
            stats[curr_behaviour]['ABL'].append(curr_bout_len)
            stats[curr_behaviour]['NB'] += 1

        i = j
    
    for lab, stat in stats.items():
        stats[lab]['TD'] = sum(stat['ABL']) / FPS
        stats[lab]['ABL'] = sum(stat['ABL'])/len(stat['ABL']) if len(stat['ABL']) > 0 else 0

        stats[lab]['ABL'] /= FPS

    return stats    

def transition_matrix_from_assay(labels):
    n_lab = get_max_label() + 1
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

def get_usage_for_strain(label_info_file, strain=None):
    with open(label_info_file, 'rb') as f:
        label_info = joblib.load(f)
    N = len(label_info['Strain'])
    
    if strain is not None:
        labels = [label_info['Labels'][i] for i in range(N) if label_info['Strain'][i] == strain]
    else:
        labels = label_info['Labels']
    del label_info
    
    usage_data = Parallel(n_jobs=-1)(delayed(behaviour_proportion)(labs) for labs in labels)
#     usage_data = np.vstack([behaviour_proportion(labs, max_label) for labs in labels])
    usage_data = np.vstack(usage_data)
    usage_data = usage_data.sum(axis=0)/usage_data.shape[0]
    return usage_data
    
def get_tmat_for_strain(label_info_file, strain=None):
    with open(label_info_file, 'rb') as f:
        label_info = joblib.load(f)
    N = len(label_info['Strain'])
    
    if strain is not None:
        labels = [label_info['Labels'][i] for i in range(N) if label_info['Strain'][i] == strain]
    else:
        labels = label_info['Labels']
    del label_info
    
#     tmat_data = [transition_matrix_from_assay(labs, max_label) for labs in labels]
    tmat_data = Parallel(n_jobs=-1)(delayed(transition_matrix_from_assay)(labs) for labs in labels)
            
    tmat = tmat_data[0]
    for i in range(1, len(tmat_data)):
        tmat += tmat_data[i]
    
    tmat = tmat / len(tmat_data)
    return tmat
    
def behaviour_proportion(labels):
    n_lab = get_max_label() + 1
    
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

def all_behaviour_info_for_all_strains(label_info_file: str):
    with open(label_info_file, 'rb') as f:
        label_info = joblib.load(f)
    N = len(label_info['Strain'])

    info = {}
    for idx in range(get_max_label() + 1):
        info[idx_2_behaviour(idx, diff=True)] = {
            'Strain': [], 
            'Sex': [], 
            'Total Duration':  [], 
            'Average Bout Length': [], 
            'No. of Bouts': []
        }
    
    for i in tqdm(range(N)):
        stats = get_stats_for_all_labels(label_info['Labels'][i])
        for idx in range(get_max_label() + 1):
            behaviour = idx_2_behaviour(idx, diff=True)
            info[behaviour]['Strain'].append(label_info['Strain'][i])
            info[behaviour]['Sex'].append(label_info['Sex'][i])
            info[behaviour]['Total Duration'].append(stats[behaviour]['TD'])
            info[behaviour]['Average Bout Length'].append(stats[behaviour]['ABL'])
            info[behaviour]['No. of Bouts'].append(stats[behaviour]['NB'])

    for behaviour, stats in info.items():
        info[behaviour] = pd.DataFrame.from_dict(stats)
    
    return info

def behaviour_usage_across_strains(label_info_file):
    with open(label_info_file, 'rb') as f:
        label_info = joblib.load(f)

    strainwise_data = {"Strain": label_info["Strain"]}
    prop = np.vstack([behaviour_proportion(l) for l in label_info["Labels"]])

    for i in range(prop.shape[1]):
        strainwise_data[idx_2_behaviour(i, diff=True)] = prop[:,i]
    
    return pd.DataFrame.from_dict(strainwise_data)

def behaviour_usage_across_strains_old(label_info_file):
    with open(label_info_file, 'rb') as f:
        label_info = joblib.load(f)
    N = len(label_info['Strain'])

    behaviour_usage = {
        'Behaviour': [],
        'Strain': [],
        'Usage': []
    }

    for i in range(N):
        usage = behaviour_proportion(label_info['Labels'][i])

        for behaviour, idxs in BEHAVIOUR_LABELS.items():
            behaviour_usage['Behaviour'].append(behaviour)
            behaviour_usage['Strain'].append(label_info['Strain'][i])
            behaviour_usage['Usage'].append(usage[idxs].sum ())
    
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

def autocorrelation(tmax, n=1000, pca=True, pca_dim=10, shuffle=False, shuffle_block_len=None):
    bsoid = BSOID.load_config('D:/IIT/DDP/data', 'dis')

    tmax = tmax * bsoid.fps // 1000
    feats = bsoid.load_features(collect=False)

    def parallel_autocorr(feats, *args):
        tmax, n, pca, pca_dim, shuffle, shuffle_block_len = args
        X = StandardScaler().fit_transform(feats)
        
        if pca:
            X = PCA(n_components=pca_dim).fit_transform(X)

        if shuffle:
            shuffle_block_len = shuffle_block_len * bsoid.fps // 1000
            # shuffle in blocks
            blocks = [X[i:i+shuffle_block_len,:] for i in range(0, X.shape[0], shuffle_block_len)]
            random.shuffle(blocks)
            X = np.vstack(blocks)
        
        res = []
        for k in np.arange(tmax):
            idxs = np.random.randint(0, X.shape[0] - k, n)
            val = sum([np.correlate(X[idx,:], X[idx+k,:]) / np.correlate(X[idx,:], X[idx,:]) for idx in idxs])
            res.append(val / n)
    
        return np.array(res)
    
    args = [tmax, n, pca, pca_dim, shuffle, shuffle_block_len]
    autocorr = Parallel(n_jobs=-1)(delayed(parallel_autocorr)(fts, *args) for fts in feats)
    return np.hstack(autocorr).T

                
def block_shuffle_autocorrelation(block_lens, tmax, n=1000):
    autocorr = [autocorrelation(
                    tmax, 
                    n, 
                    pca=False, 
                    pca_dim=None, 
                    shuffle=True, 
                    shuffle_block_len=block_lens[i]
                ).mean(axis=0) for i in tqdm(range(len(block_lens)))]
    return np.array(autocorr)

def group_stats(stats_file):
    with open(stats_file, "rb") as f:
        stats = joblib.load(f)
    
    check_along = ["Sex", "Strain"]
    grouped_stats = {}

    for idx in range(get_max_label() + 1):
        grp_label, diff_label = idx_2_behaviour(idx, diff=False), idx_2_behaviour(idx, diff=True)

        if grp_label not in grouped_stats:
            grouped_stats[grp_label] = stats[diff_label]
        else:
            df = grouped_stats[grp_label]
            assert not (df[check_along] != stats[diff_label][check_along]).any().any(), "(FATAL) mismatch in order when grouping"

            for metric in ["Total Duration", "No. of Bouts"]:
                df[metric] += stats[diff_label][metric]
            
            df["Average Bout Length"] = [x / y if y > 0 else 0 for x, y in zip(df["Total Duration"], df["No. of Bouts"])]
            
    return grouped_stats

def calculate_PVE(stats, metric):
    PVE = {}
    for i in range(get_max_label() + 1):
        behaviour = idx_2_behaviour(i, diff=True)
        strainwise_data = {}

        N, T, zbar = 0, 0, 0
        for i, strain in enumerate(stats[behaviour]['Strain']):
            if strain in strainwise_data:
                strainwise_data[strain].append(stats[behaviour][metric][i])
            else:
                strainwise_data[strain] = [stats[behaviour][metric][i]]
                N += 1
            T += 1
            zbar += stats[behaviour][metric][i]
        zbar /= T

        SS_s, SS_e, n0 = 0, 0, 0
        for _, data in strainwise_data.items():
            data = np.array(data)
            SS_s += data.size * ((data.mean() - zbar) ** 2)
            SS_e += data.size * data.var()
            n0 += data.size ** 2

        MS_s = SS_s / (N - 1)
        MS_e = SS_e / (T - N)
        n0 = (T - (n0 / T)) / (N - 1)

        Var_s = (MS_s - MS_e) / n0
        Var_z = Var_s + MS_e

        pve = Var_s / Var_z
        se_pve = (2 * ((1 - pve) ** 2) * ((1 + (n0 - 1) * pve) ** 2)) / (N * n0 * (n0 - 1))
        PVE[behaviour] = (pve, 0.5 * (se_pve ** 0.5))
    
    return PVE

def GEMMA_csv_input(label_info_file, input_csv):
    label_info = joblib.load(open(label_info_file, 'rb'))
    fields = ('Strain', 'Sex', 'MouseID', 'NetworkFilename')
    while len(label_info['Strain']) > 0:
        key = [label_info[f][0] + ':' for f in fields]
        key = ''.join(key)[:-1]
        label_info[key] = label_info['Labels'][0]

        for f in fields:
            del label_info[f][0]
        del label_info['Labels'][0]
    
    if input_csv.endswith('.tsv'):
        data = pd.read_csv(input_csv, sep='\t')    
    else:
        data = pd.read_csv(input_csv)
    N = data.shape[0]

    phenotypes = []
    for lab, idxs in BEHAVIOUR_LABELS.items():
        for i in range(len(idxs)):
            phenotypes.append(f'{lab}_{i}_TD')
            phenotypes.append(f'{lab}_{i}_ABL')
            phenotypes.append(f'{lab}_{i}_NB')
    new_cols = list(data.columns)
    new_cols.extend(phenotypes)
    
    data = data.reindex(columns=new_cols)
    drop_idxs = []
    for i in tqdm(range(N)):
        metadata = dict(data.iloc[i])
        key = [metadata[f] + ':' for f in fields]
        key = ''.join(key)[:-1]

        if key in label_info:
            stats = get_stats_for_all_labels(label_info[key])
            for p in phenotypes:
                lab, idx, metric = p.split('_')
                data.at[i, p] = stats[f'{lab} #{idx}'][metric]            
        else:
            drop_idxs.append(i)

    data = data.drop(drop_idxs)
    data.to_csv('./gemma_input.csv', index=False)

def filter_strains(input_csv, droplist):
    df = pd.read_csv(input_csv)
    filt_df = df[~df["Strain"].isin(droplist)]
    filt_df.to_csv("./filt_gemma_input.csv", index=False)
    
def GEMMA_config_files():
    import yaml
    
    config = {}
    
    config["strain"] = "Strain"
    config["sex"] = "Sex"
    
    config["phenotypes"] = {}
    config["groups"] = []
    for lab, idxs in BEHAVIOUR_LABELS.items():
        for i in range(len(idxs)):
            key = f"{lab}n{i}"
            config["phenotypes"][f"{lab}_{i}_TD"] = {"papername": f"{lab}n{i}_TD", "group": lab}
            config["phenotypes"][f"{lab}_{i}_ABL"] = {"papername": f"{lab}n{i}_ABL", "group": lab}
            config["phenotypes"][f"{lab}_{i}_NB"] = {"papername": f"{lab}n{i}_NB", "group": lab}
        config["groups"].append(lab)
    
    config["covar"] = ["Sex"]

    with open('./default_GEMMA.yaml', 'r') as f:
        config.update(yaml.load(f, Loader=yaml.FullLoader))

    with open('./gemma_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    import random
    config["phenotypes"] = {"Groom_4_TD": config["phenotypes"]["Groom_4_TD"]}

    with open('./gemma_shuffle.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
def get_keypoint_data(metadata, label_info_file, data_dir):
    import pysftp
    try:
        pose_dir, _ = get_pose_data_dir(data_dir, metadata['NetworkFilename'])
        _, _, movie_name = metadata['NetworkFilename'].split('/')
        filename = f'{pose_dir}/{movie_name[0:-4]}_pose_est_v2.h5'    
        with pysftp.Connection('login.sumner.jax.org', username='laadd', password='CuppY1798CakE@') as sftp:
            sftp.get(filename)
        
        f = h5py.File(filename.split('/')[-1], 'r')
        data = list(f.keys())[0]
        keys = list(f[data].keys())
        conf, pos = np.array(f[data][keys[0]]), np.array(f[data][keys[1]])
        f.close()

        data = bsoid_format(conf, pos)
        fdata, perc_filt = likelihood_filter(data, fps=FPS, end_trim=2, clip_window=0, conf_threshold=0.3)

        strain, mouse_id = metadata['Strain'], metadata['MouseID']
        if perc_filt > 10:
            print(f'mouse:{strain}/{mouse_id}: % data filtered from raw data is too high ({perc_filt} %)')
        
        data_shape = fdata['x'].shape
        print(f'preprocessed raw data of shape: {data_shape} for mouse:{strain}/{mouse_id}')

        with open(label_info_file, "rb") as f:
            info = joblib.load(f)
        N = len(info["Strain"])
        identifiers = ("Strain", "Sex", "NetworkFilename", "MouseID")
        for i in range(N):
            match = True
            for idx in identifiers:
                if info[idx][i] != metadata[idx]:
                    match = False
            
            if match:
                break
        
        labels = info["Labels"][i]
        metadata["keypoints"] = fdata
        metadata["labels"] = labels
        
        import os
        os.remove(filename.split('/')[-1])
        
        return metadata

    except Exception as e:
        print(e)
        return None
    
# def get_transition_data(label_info_file):
    
if __name__ == "__main__":
    label_info_file = '/Users/dhruvlaad/IIT/DDP/data/analysis/label_info.pkl'
    # stats_file = base_dir + 'analysis/stats.pkl'
    # plot_dir = 'C:/Users/dhruvlaad/Desktop/plots'
    
    # behaviour_stats = all_behaviour_info_for_all_strains(label_info_file)
    # with open(stats_file, 'wb') as f:
    #     joblib.dump(behaviour_stats, f)   

    # GEMMA_csv_input(label_info_file, input_csv)
    # GEMMA_config_files()

    autocorrelation(tmax=3000, n=1000)