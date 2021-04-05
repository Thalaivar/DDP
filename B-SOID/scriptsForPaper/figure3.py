import os
import cv2
import joblib
import ftplib
import numpy as np
import pandas as pd

SAVE_DIR = "D:/IIT/DDP/data/paper/figure2"
try: os.mkdir(SAVE_DIR)
except FileExistsError: pass

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage

import sys
sys.path.insert(0, "D:/IIT/DDP/DDP/B-SOID")
from analysis import *
from BSOID.utils import get_video_and_keypoint_data
from tqdm import tqdm
from sklearn.cluster import SpectralClustering

def plot_transition_matrix(label_info_file: str):
    tmat = get_tmat_for_strain(label_info_file)

    labels = SpectralClustering(affinity='precomputed', n_clusters=4).fit_predict(tmat)
    spectral_tmat = np.zeros_like(tmat)
    groups = [[] for x in range(labels.max()+1)]
    for i, x in enumerate(labels):
        groups[x].append(i)

    idx_labels, new_idx_map, i = [], {}, 0
    for g in groups:
        for idx in g:
            idx_labels.append(idx_2_behaviour(idx, diff=True))
            new_idx_map[idx] = i
            i += 1

    for i, g in enumerate(groups):
        for idx1 in g:
            for idx2 in g:
                spectral_tmat[new_idx_map[idx1], new_idx_map[idx2]] = tmat[idx1, idx2]
        
        for j, h in enumerate(groups):
            if i != j:
                for idx1 in g:
                    for idx2 in h:
                        spectral_tmat[new_idx_map[idx1], new_idx_map[idx2]] = tmat[idx1, idx2]
    
    fig, ax = plt.subplots(figsize=(9,9))
    ax.patch.set_facecolor('white')
    ax.set_aspect('equal', 'box')
    ax.set_xticks(np.arange(20))
    ax.set_yticks(np.arange(20))
    ax.set_xticklabels(idx_labels)
    ax.set_yticklabels(idx_labels)
    ax.tick_params(axis='x', labelrotation=90, labelsize=12)
    ax.tick_params(axis='y', labelrotation=0, labelsize=12)

    max_weight = 0.25
    for (x, y), w in np.ndenumerate(spectral_tmat):
        color = (0.1, 0.44, 0.82)
        if w > max_weight:
            color = (0.96, 0.26, 0.71)
            w = max_weight / 6
        size = np.sqrt(np.abs(w) / max_weight)
        rect = patches.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

    sz = 20
    b1, = ax.plot([], marker='s', markersize=9.5, linestyle='', color = (0.96, 0.26, 0.71), label=r'$p > 0.25$')
    b2, = ax.plot([], marker='s', markersize=sz, linestyle='', color=(0.1, 0.44, 0.82), label=r'$p = 0.25$')

    leg = plt.legend(handles=[b1, b2], bbox_to_anchor=(0.5, 1.05), loc='upper left', prop={'size': 14}, ncol=2)
    leg.get_frame().set_linewidth(0.0)

    sns.despine(trim=True)
    # plt.savefig(f'{plot_dir}/tmat.png', dpi=600, bbox_inches='tight')
    plt.show()

plot_transition_matrix(label_info_file="../../data/analysis/label_info.pkl")