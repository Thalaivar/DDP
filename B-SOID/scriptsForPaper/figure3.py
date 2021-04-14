import os
import cv2
import joblib
import ftplib
import numpy as np
import pandas as pd
import networkx as nx

# base_dir = "D:/IIT/DDP"
base_dir = "/Users/dhruvlaad/IIT/DDP/"
SAVE_DIR = os.path.join(base_dir, "data/paper/figure3")

try: os.mkdir(SAVE_DIR)
except FileExistsError: pass

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage

import sys
sys.path.insert(0, os.path.join(base_dir, "DDP/B-SOID/"))
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
    plt.savefig(os.path.join(SAVE_DIR, "tmat.png"), dpi=400, bbox_inches='tight', pad_inches=0)
    plt.show()

def bin_wts(wt, bins):
    if bins[0] >= 0:
        for i in range(len(bins)-1):
            if wt >= bins[i] and wt < bins[i+1]:
                return i
        return len(bins) - 1
    else:
        if wt < bins[0]:
            return 0
        for i in range(0, len(bins)-1):
            if wt >= bins[i] and wt < bins[i+1]:
                return i + 1
        return len(bins)

def tnet_from_tmat(tmat, behaviour_usage, diff_graph=False):        
    if diff_graph:
        bins = [-0.05, -0.025, -0.01, 0, 0.01, 0.025, 0.05]
    else:
        bins = [0, 0.025, 0.075, 0.15]

    for i in range(tmat.shape[0]):
        tmat[i,i] = -1

    G = nx.MultiDiGraph()
    G.add_nodes_from(range(0, tmat.shape[0]))

    not_allowed = [0] if not diff_graph else [3, 4]
    for i in range(tmat.shape[0]):
        for j in range(tmat.shape[1]):
            if i != j:
                bin_idx = bin_wts(tmat[i,j], bins)
                if bin_idx not in not_allowed:
                    G.add_edge(i, j, weight=tmat[i,j])
    if diff_graph:
        edge_wts = [5, 1, 0.2, 0, 0, 0.2, 1, 5]
    else:
        edge_wts = [0, 0.2, 1, 5]
    widths = [edge_wts[bin_wts(G[u][v][0]['weight'], bins)] for u, v in G.edges()]
    return G, widths, bins

def behavioral_statemap(label_info_file, strain, pos=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    tmat = get_tmat_for_strain(label_info_file, strain)
    usage = get_usage_for_strain(label_info_file, strain)
    G, widths, bins = tnet_from_tmat(tmat, usage, diff_graph=False)
    neg_color = tuple([x/255 for x in [189, 27, 15, 0.85 * 255]])
    pos_color = tuple([x / 255 for x in [18 ,63, 161 ,0.85 * 255]])
    
    strain = strain.replace("/", "-")
    if pos is None:
        pos = nx.spectral_layout(G)
    nx.draw(G, ax=ax, pos=pos, node_size=2000*usage, connectionstyle='bar, fraction = 0.01', edge_color=pos_color, edgecolors='k', node_color='white', width=widths, arrowsize=5)
    plt.savefig(os.path.join(SAVE_DIR, f"{strain}_statemap.png"), dpi=400, bbox_inches='tight', pad_inches=0)
    plt.show()

    return G

def diff_statemap_for_strains(label_info_file, strains, pos=None):
    usage = get_usage_for_strain(label_info_file, strains[0]) - get_usage_for_strain(label_info_file, strains[1])
    tmat = get_tmat_for_strain(label_info_file, strains[0]) - get_tmat_for_strain(label_info_file, strains[1])
    
    fig, ax = plt.subplots(figsize=(12, 12))
    neg_color = tuple([x/255 for x in [189, 27, 15, 0.85 * 255]])
    pos_color = tuple([x / 255 for x in [18 ,63, 161 ,0.85 * 255]])
    G, widths, bins = tnet_from_tmat(tmat, usage, diff_graph=True)
    node_colors = []
    for i in range(usage.shape[0]):
        if usage[i] < 0:
            node_colors.append(neg_color)
        else:
            node_colors.append(pos_color)
    edge_colors = []
    for u, v in G.edges():
        if G[u][v][0]['weight'] >= 0:
            edge_colors.append(pos_color)
        else:
            edge_colors.append(neg_color)
    
    pos = nx.spectral_layout(G) if pos is None else pos
    nx.draw(
            G, 
            ax=ax,
            pos=pos, 
            node_size=2000*usage,
            connectionstyle='bar, fraction = 0.01', 
            edge_color=edge_colors, 
            edgecolors=node_colors, 
            node_color='white', 
            width=widths,
            arrowsize=5
        )

    plt.show()

    return G

def statemap_scatter_plot(strain_list, pos=None):
    data = {}
    for i in tqdm(range(len(strain_list))):
        data[strain_list[i]] = (
                    get_usage_for_strain(label_info_file, strain_list[i]), 
                    get_tmat_for_strain(label_info_file, strain_list[i])
                )
    
    N = len(strain_list) * (len(strain_list) - 1) / 2
    N = int(N ** 0.5)
    fig, ax = plt.subplots(N, N, figsize=(12,12))
    neg_color = tuple([x/255 for x in [189, 27, 15, 0.85 * 255]])
    pos_color = tuple([x / 255 for x in [18 ,63, 161 ,0.85 * 255]])
    
    usage = [data[s1][0] - data[s2][0] for s1, s2 in combinations(strain_list, 2)]
    tmat = [data[s1][1] - data[s2][1] for s1, s2 in combinations(strain_list, 2)]
    k = 0
    for i in range(N):
        for j in range(N):
            diff_statemap_for_strains(usage[k], tmat[k], ax[i][j], pos)
            k += 1
    
    plt.show()

if __name__ == "__main__":
    label_info_file = "../../data/analysis/label_info.pkl"
    # plot_transition_matrix(label_info_file="../../data/analysis/label_info.pkl")
    
    strains = ["NZO/HILtJ", "MSM/MsJ"]
    G = behavioral_statemap(label_info_file, strain="C57BL/6NJ")
    G1 = behavioral_statemap(label_info_file, strains[0], pos=nx.spectral_layout(G))
    G2 = behavioral_statemap(label_info_file, strains[1], pos=nx.spectral_layout(G))
    G3 = diff_statemap_for_strains(label_info_file, strains, pos=nx.spectral_layout(G))