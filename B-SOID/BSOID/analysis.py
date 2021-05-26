import os
import h5py
import psutil
from joblib.parallel import Parallel, delayed
import yaml
import numpy as np
import pandas as pd

import logging
from BSOID import data
logger = logging.getLogger(__name__)

from BSOID.data import *
from BSOID.preprocessing import likelihood_filter
from BSOID.features import extract_comb_feats, aggregate_features
from prediction import get_frameshifted_prediction

def get_data(metadata, data_dir, fps, min_video_len):
    try:
        pose_dir, _ = get_pose_data_dir(data_dir, metadata["NetworkFilename"])
        _, _, movie_name = metadata['NetworkFilename'].split('/')
        filename = f'{pose_dir}/{movie_name[0:-4]}_pose_est_v2.h5'

        conf, pos = process_h5py_data(h5py.File(filename, 'r'))
        if conf.shape[0] >= min_video_len:
            data = bsoid_format(conf, pos)
            fdata, perc_filt = likelihood_filter(data, fps, end_trim=5, clip_window=-1, conf_threshold=0.3)

            strain, mouse_id = metadata['Strain'], metadata['MouseID']
            if perc_filt > 10:
                logging.warning(f'mouse:{strain}/{mouse_id}: % data filtered from raw data is too high ({perc_filt} %)')
        
            data_shape = fdata['x'].shape
            logging.info(f'preprocessed raw data of shape: {data_shape} for mouse:{strain}/{mouse_id}')

            return fdata
    except Exception as e:
        logger.warning(e)
    
    return None

def labels_for_mouse(metadata, clf, data_dir, fps, stride_window):
    fdata = get_data(metadata, data_dir, fps)
    if fdata is not None:
        labels = get_frameshifted_prediction(fdata, clf, fps, stride_window)
        return labels
    else:
        return None

def bout_stats(labels, max_label, min_bout_len, fps):
    stats = {}
    for i in range(max_label):
        stats[i] = {'td': None, 'abl': [], 'nb': 0}
    
    i = 0
    while i < len(labels) - 1:
        curr_label = labels[i]
        curr_bout_len, j = 1, i + 1

        while (j < len(labels) - 1) and (labels[j] == curr_label):
            curr_label = labels[j]
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
    tmat = np.zeros((max_label, max_label))
    
    curr_lab = labels[0]
    for i in range(1, labels.size):
        tmat[curr_lab, labels[i]] += 1
        curr_lab = labels[i]
    
    tmat = tmat / tmat.sum(axis=1, keepdims=True)
    return tmat

def proportion_usage(labels, max_label):
    _, prop = np.unique(labels, return_counts=True)
    prop /= labels.size
    return prop

def extract_labels(input_csv, data_dir, clf, fps, stride_window):
    N = input_csv.shape[0]

    def par_labels(metadata, clf, data_dir, fps, stride_window):
        labels = labels_for_mouse(metadata, clf, data_dir, fps, stride_window)
        return [metadata, labels]

    label_data = Parallel(n_jobs=psutil.cpu_count(logical=False))(delayed(par_labels)(input_csv.iloc[i], clf, data_dir, fps, stride_window) for i in range(N))
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

def bout_stats_for_all_strains(label_info, max_label, min_bout_len, fps):
    behavioral_bout_stats = {}

    strains = list(label_info.keys())
    for i in tqdm(range(len(strains))):
        _, data = strains[i], label_info[strains[i]]
        for d in data:
            key = get_key_from_metadata(d["metadata"])
            behavioral_bout_stats[key] = {}

            stats = bout_stats(d["labels"], max_label, min_bout_len, fps)
            for i in range(max_label):
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
            {"metadata": d["metadata"], "tmat": transition_matrix(d["labels"], max_label)}
            for d in data
        ]
    
    return tmat

def gemma_files(label_info, input_csv, max_label, min_bout_len, fps, default_config_file):
    strain_bout_stats = bout_stats_for_all_strains(label_info, max_label, min_bout_len, fps)
    
    stats_df = {f"phenotype_{i}_td": [] for i in range(max_label)}
    stats_df.update({f"phenotype_{i}_abl": [] for i in range(max_label)})
    stats_df.update({f"phenotype_{i}_nb": [] for i in range(max_label)})

    retain_idx = []
    for i in range(input_csv.shape[0]):
        key = get_key_from_metadata(dict(input_csv.iloc[i]))
        if key in strain_bout_stats:
            for i in range(max_label):
                stats_df[f"phenotype_{i}_td"].append(strain_bout_stats[key][f"phenotype_{i}_td"])
                stats_df[f"phenotype_{i}_abl"].append(strain_bout_stats[key][f"phenotype_{i}_abl"])
                stats_df[f"phenotype_{i}_nb"].append(strain_bout_stats[key][f"phenotype_{i}_nb"])
            retain_idx.append(i)
    
    gemma_csv = pd.concat([input_csv.iloc[retain_idx, :], pd.DataFrame.from_dict(stats_df)], axis=1)

    config = {}
    config["strain"] = "Strain"
    config["sex"] = "Sex"
    config["phenotypes"] = {}
    config["groups"] = ["Total Duration", "Average Bout Length", "Number of Bouts"]

    for i in range(max_label):
        config["phenotypes"][f"phenotype_{i}_TD"] = {"papername": f"phenotype_{i}_TD", "group": "Total Duration"}
        config["phenotypes"][f"phenotype_{i}_ABL"] = {"papername": f"phenotype_{i}_ABL", "group": "Average Bout Length"}
        config["phenotypes"][f"phenotype_{i}_NB"] = {"papername": f"phenotype_{i}_NB", "group": "Number of Bouts"}

    config["covar"] = ["Sex"]

    with open(default_config_file, 'r') as f:
        config.update(yaml.load(f, Loader=yaml.FullLoader))
    
    config_file = os.path.join(os.path.split(default_config_file)[0], "gemma_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    gemma_csv.to_csv(os.path.join(os.path.split(default_config_file)[0], "gemma_data.csv"))
