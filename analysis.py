import os
import h5py
import psutil
from joblib.parallel import Parallel, delayed
import yaml
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

from BSOID.data import *
from BSOID.preprocessing import likelihood_filter
from BSOID.features import extract_comb_feats, aggregate_features
from prediction import get_frameshifted_prediction

import scipy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_data(metadata, data_dir, bsoid, min_video_len):
    try:
        pose_dir, _ = get_pose_data_dir(data_dir, metadata["NetworkFilename"])
        _, _, movie_name = metadata['NetworkFilename'].split('/')
        filename = f'{pose_dir}/{movie_name[0:-4]}_pose_est_v2.h5'

        conf, pos = process_h5py_data(h5py.File(filename, 'r'))
        if conf.shape[0] >= min_video_len:
            data = bsoid_format(conf, pos)
            fdata, perc_filt = likelihood_filter(data, bsoid.fps, bsoid.conf_threshold, bsoid.bodyparts, end_trim=5, clip_window=-1)

            strain, mouse_id = metadata['Strain'], metadata['MouseID']
            if perc_filt > 10:
                logging.warning(f'mouse:{strain}/{mouse_id}: % data filtered from raw data is too high ({perc_filt} %)')
        
            data_shape = fdata['x'].shape
            logging.info(f'preprocessed raw data of shape: {data_shape} for mouse:{strain}/{mouse_id}')

            return fdata
    except Exception as e:
        logger.warning(e)
    
    return None

def labels_for_mouse(metadata, clf, data_dir, bsoid, min_video_len):
    fdata = get_data(metadata, data_dir, bsoid, min_video_len)
    if fdata is not None:
        labels = get_frameshifted_prediction(fdata, clf, bsoid.fps, bsoid.stride_window)
        return labels.astype(int)
    else:
        return None

def bout_stats(labels, max_label, min_bout_len, fps, inv_group_map=None):
    if inv_group_map is None:
        inv_group_map = {i: i for i in range(max_label)}

    stats = {}
    min_bout_len = min_bout_len * fps // 1000

    for i in range(max_label):
        stats[inv_group_map[i]] = {'td': None, 'abl': [], 'nb': 0}

    i = 0
    while i < len(labels) - 1:
        curr_label = inv_group_map[labels[i]]
        curr_bout_len, j = 1, i + 1
        while (j < len(labels) - 1) and (inv_group_map[labels[j]] == curr_label):
            curr_label = inv_group_map[labels[j]]
            curr_bout_len += 1
            j += 1
            
        if curr_bout_len >=  min_bout_len:
            stats[curr_label]['abl'].append(curr_bout_len)
            stats[curr_label]['nb'] += 1

        i = j
    
    for lab, stat in stats.items():
        stats[lab]['td'] = sum(stat['abl']) / fps
        stats[lab]['abl'] = sum(stat['abl'])/len(stat['abl']) if len(stat['abl']) > 0 else 0

        stats[lab]['abl'] /= fps

    return stats    

def transition_matrix(labels, max_label):
    labels = labels.astype(int)
    
    tmat = np.zeros((max_label, max_label))
    
    curr_lab = labels[0]
    for i in range(1, labels.size):
        tmat[curr_lab, labels[i]] += 1
        curr_lab = labels[i]
    
    for i in range(tmat.shape[0]):
        if tmat[i].sum() > 0:
            tmat[i] /= tmat[i].sum()

    return tmat

def proportion_usage(labels, max_label):
    labels = labels.astype(int)

    prop = np.zeros((max_label,))
    class_labels, counts = np.unique(labels, return_counts=True)

    for i in range(class_labels.size):
        prop[class_labels[i]] = counts[i]
        
    prop /= labels.size
    return prop

def extract_labels(input_csv, data_dir, clf, bsoid, min_video_len):
    N = input_csv.shape[0]

    def par_labels(metadata, clf, data_dir, bsoid, min_video_len):
        labels = labels_for_mouse(metadata, clf, data_dir, bsoid, min_video_len)
        return [metadata, labels]

    label_data = Parallel(n_jobs=psutil.cpu_count(logical=False))(delayed(par_labels)(input_csv.iloc[i], clf, data_dir, bsoid, min_video_len) for i in tqdm(range(N)))
    label_data = [l for l in label_data if l[1] is not None]
    
    strain_labels = {}
    for (metadata, labels) in label_data:
        strain = metadata["Strain"]
        if strain in strain_labels:
            strain_labels[strain].append({"metadata": metadata, "labels": labels})
        else:
            strain_labels[strain] = [{"metadata": metadata, "labels": labels}]
    
    logger.info(f"extracted labels from {len(label_data)} mice from {len(strain_labels)} strains")
    return strain_labels

def get_key_from_metadata(metadata):
    fields = ("MouseID", "Sex", "Strain", "NetworkFilename")
    key = "".join([f"{metadata[f]};" for f in fields])
    key = key[:-1]
    return key

def bout_stats_for_all_strains(label_info, max_label, min_bout_len, fps, inv_group_map=None):
    behavioral_bout_stats = {}

    strains = list(label_info.keys())
    for i in tqdm(range(len(strains))):
        _, data = strains[i], label_info[strains[i]]
        for d in data:
            key = get_key_from_metadata(d["metadata"])
            behavioral_bout_stats[key] = {}

            stats = bout_stats(d["labels"], max_label, min_bout_len, fps, inv_group_map)
            for i in stats.keys():
                behavioral_bout_stats[key][f"phenotype_{i}_td"] = stats[i]["td"]
                behavioral_bout_stats[key][f"phenotype_{i}_abl"] = stats[i]["abl"]
                behavioral_bout_stats[key][f"phenotype_{i}_nb"] = stats[i]["nb"]

    return behavioral_bout_stats

def proportion_usage_across_strains(label_info, max_label):
    prop = {}
    for strain, data in label_info.items():
        prop[strain] = [
            {"metadata": d["metadata"], "prop": proportion_usage(d["labels"], max_label)}
            for d in data
        ]
    
    return prop

def transition_matrix_across_strains(label_info, max_label):
    tmat = {}
    for strain, data in label_info.items():
        tmat[strain] = [
            {"metadata": d["metadata"], "tmat": transition_matrix(d["labels"].astype(int), max_label)}
            for d in data
        ]
    
    return tmat

def gemma_transitions(label_info, input_csv, max_label, default_config_file, thresh=None, exclude_strains=None, nmax=10):
    tmats = transition_matrix_across_strains(label_info, max_label)

    if exclude_strains is not None:
        exclude_strains = []

    transitions = {}
    for _, data in tmats.items():
        for d in data:
            transitions[get_key_from_metadata(d["metadata"])] = d["tmat"]
    del tmats

    trans_df = {}
    for i in range(max_label):
        for j in range(max_label):
            trans_df[f"transition_{i}_{j}"] = []
    
    retain_idx = []
    for k in range(input_csv.shape[0]):
        key = get_key_from_metadata(dict(input_csv.iloc[k]))
        if key in transitions:
            for i in range(max_label):
                for j in range(max_label):
                    trans_df[f"transition_{i}_{j}"].append(transitions[key][i,j])
    
            retain_idx.append(k)
            
    transitions_csv = pd.concat([input_csv.iloc[retain_idx, :].reset_index(), pd.DataFrame.from_dict(trans_df)], axis=1)
    
    downsampled_dfs = []
    for strain, df in transitions_csv.groupby("Strain"):
        if strain not in exclude_strains:
            if df.shape[0] > nmax:
                downsampled_dfs.append(df.sample(n=nmax))
            else:
                downsampled_dfs.append(df)
    transitions_csv = pd.concat(downsampled_dfs, axis=0)

    if thresh is not None:
        # get no. of unique values in each transition
        utrans = transitions_csv.nunique(axis=0)
        drop_cols = [utrans.index[i] for i in range(utrans.size) if "transition" not in utrans.index[i]]
        utrans = utrans.drop(drop_cols)

        utrans = utrans.sort_values(ascending=False)
        top_trans = list(utrans.index[:round(thresh * utrans.size)])

        for col in transitions_csv.columns:
            if (col not in top_trans) and ("transition" in col):
                transitions_csv.drop(col, axis=1, inplace=True)

    transitions_csv.drop("TestAssayNotes", axis=1, inplace=True)
    
    config = {}
    config["strain"] = "Strain"
    config["sex"] = "Sex"
    config["phenotypes"] = {}
    config["groups"] = ["Transitions"]

    for col in transitions_csv.columns:
        if "transition" in col:
            _, i, j = col.split('_')
            i, j = int(i), int(j)
            config["phenotypes"][col] = {"papername": f"T{i}-{j}", "group": "Transitions"}
    
    config["covar"] = ["Sex"]

    with open(default_config_file, 'r') as f:
        config.update(yaml.load(f, Loader=yaml.FullLoader))
    
    config_file = os.path.join(os.path.split(default_config_file)[0], "gemma_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    transitions_csv.to_csv(os.path.join(os.path.split(default_config_file)[0], "gemma_trans.csv"), index=False)

def gemma_files(label_info, input_csv, max_label, min_bout_len, fps, default_config_file, nmax=10, exclude_strains=None, inv_group_map=None):
    strain_bout_stats = bout_stats_for_all_strains(label_info, max_label, min_bout_len, fps, inv_group_map)
    
    stats_df = {f"phenotype_{i}_td": [] for i in range(max_label)}
    stats_df.update({f"phenotype_{i}_abl": [] for i in range(max_label)})
    stats_df.update({f"phenotype_{i}_nb": [] for i in range(max_label)})

    retain_idx = []
    for k in range(input_csv.shape[0]):
        key = get_key_from_metadata(dict(input_csv.iloc[k]))
        if key in strain_bout_stats:
            for i in range(max_label):
                stats_df[f"phenotype_{i}_td"].append(strain_bout_stats[key][f"phenotype_{i}_td"])
                stats_df[f"phenotype_{i}_abl"].append(strain_bout_stats[key][f"phenotype_{i}_abl"])
                stats_df[f"phenotype_{i}_nb"].append(strain_bout_stats[key][f"phenotype_{i}_nb"])
            retain_idx.append(k)
    
    gemma_csv = pd.concat([input_csv.iloc[retain_idx, :].reset_index(), pd.DataFrame.from_dict(stats_df)], axis=1)
    
    if exclude_strains is None:
        exclude_strains = []
    
    downsampled_dfs = []
    for strain, df in gemma_csv.groupby("Strain"):
        if strain not in exclude_strains:
            if df.shape[0] > nmax:
                downsampled_dfs.append(df.sample(n=nmax))
            else:
                downsampled_dfs.append(df)
    gemma_csv = pd.concat(downsampled_dfs, axis=0)
    
    config = {}
    config["strain"] = "Strain"
    config["sex"] = "Sex"
    config["phenotypes"] = {}
    config["groups"] = ["Total Duration", "Average Bout Length", "Number of Bouts"]

    for i in range(max_label):
        config["phenotypes"][f"phenotype_{i}_td"] = {"papername": f"phenotype_{i}_TD", "group": "Total Duration"}
        config["phenotypes"][f"phenotype_{i}_abl"] = {"papername": f"phenotype_{i}_ABL", "group": "Average Bout Length"}
        config["phenotypes"][f"phenotype_{i}_nb"] = {"papername": f"phenotype_{i}_NB", "group": "Number of Bouts"}

    config["covar"] = ["Sex"]

    with open(default_config_file, 'r') as f:
        config.update(yaml.load(f, Loader=yaml.FullLoader))
    
    config_file = os.path.join(os.path.split(default_config_file)[0], "gemma_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    gemma_csv.to_csv(os.path.join(os.path.split(default_config_file)[0], "gemma_data.csv"), index=False)

def get_group_map(eac_mat_file, pvalue=None, plot=False):
    eac_mat = np.load(eac_mat_file)
    Z = scipy.cluster.hierarchy.linkage(squareform(eac_mat), method="single")
    
    if pvalue is None:
        plot = True
        fig, ax = plt.subplots(figsize=(6, 9))
        ddata = scipy.cluster.hierarchy.dendrogram(Z, ax=ax)
        ylocs = []
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            ax.plot([0, x], [y, y], '--g')
            ylocs.append(y)

        plt.show()

        pvalue = float(input("p-value for choosing split: "))
    
    group_labels = scipy.cluster.hierarchy.fcluster(Z, t=pvalue, criterion="distance")
    
    if plot:
        fig, ax = plt.subplots(figsize=(6, 9))
        cmap = plt.cm.tab20b(group_labels)
        scipy.cluster.hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
        scipy.cluster.hierarchy.dendrogram(Z, ax=ax, color_threshold=pvalue)
        xlim = ax.get_xlim()
        ax.plot([0, xlim[1]], [pvalue, pvalue], '--r')
        ax.set_xticks([])
        ax.set_xticklabels([])
        plt.show()

    group_labels = scipy.cluster.hierarchy.fcluster(Z, t=pvalue, criterion="distance")
    
    group_map = {lab: [] for lab in np.unique(group_labels)}
    inv_group_map = {}
    for i, lab in enumerate(group_labels):
        group_map[lab].append(i)
        inv_group_map[i] = lab
    
    group_labels = {
        1: "Locomotion #1",
        2: "Locomotion #2",
        3: "Head Sweeps",
        4: "Grooming",
        5: "Sniffing + Grooming",
        6: "Head Sweeps + Turning",
        7: "Rearing",
        8: "Rearing + Sniffing",
        9: "Reared Grooming #1",
        10: "Turning",
        11: "Stationary Grooming",
        12: "Posterior Grooming #1",
        13: "Posterior Grooming #2",
        14: "Paused",
        15: "Reared Grooming #2" 
    }

    return group_map, inv_group_map, group_labels