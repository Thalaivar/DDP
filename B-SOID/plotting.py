from seaborn.external.husl import lch_to_husl
from BSOID.features.displacement_feats import window_extracted_feats
from BSOID.preprocessing import windowed_feats
from itertools import combinations, permutations
import math
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from analysis import *
from joblib import delayed, Parallel

base_dir = '/Users/dhruvlaad/IIT/DDP/data/'
# base_dir = 'D:/IIT/DDP/data/'
label_info_file = base_dir + 'analysis/label_info.pkl'
stats_file = base_dir + 'analysis/stats.pkl'
# plot_dir = 'C:/Users/dhruvlaad/Desktop/plots'
plot_dir = './plots'

# with open(stats_file, 'rb') as f:
#     behaviour_stats = joblib.load(f)
# behaviour_stats = all_behaviour_info_for_all_strains(label_info_file)
# with open(stats_file, 'wb') as f:
#     joblib.dump(behaviour_stats, f)   

########################################################################################################################
#                                                    Transition Networks                                               #
########################################################################################################################
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

def behavioural_statemap_for_strain(label_info_file, strain=None, draw=True):
    fig, ax = plt.subplots(figsize=(12, 12))
    tmat = get_tmat_for_strain(label_info_file, strain)
    usage = get_usage_for_strain(label_info_file, strain)
    G, widths, bins = tnet_from_tmat(tmat, usage, diff_graph=False)
    neg_color = tuple([x/255 for x in [189, 27, 15, 0.85 * 255]])
    pos_color = tuple([x / 255 for x in [18 ,63, 161 ,0.85 * 255]])
    if draw:
        nx.draw(G, ax=ax, pos=nx.spectral_layout(G), node_size=2000*usage, connectionstyle='bar, fraction = 0.01', edge_color=pos_color, edgecolors='k', node_color='white', width=widths, arrowsize=5)
        plt.show()
    return G

def diff_statemap_for_strains(usage, tmat, ax, pos=None):
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
    return ax

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

def plot_keypoint_data(data, behaviours=None, limits=None):
    idx2grp = idx2group_map()
    labels = [idx2grp[l] for l in data["labels"]]
    x, y = data["keypoints"]['x'][:len(labels),:], data["keypoints"]['y'][:len(labels),:]
    
    behaviours = [idx2grp[x] for x in behaviours]

    if limits is None:
        limits = [0, len(labels)]

    i, maxlen = limits[0], [-1, None, None]
    while i < limits[1]:
        curr_label = labels[i]
        if curr_label in behaviours:
            j = i + 1
            while j < limits[1] and labels[j] in behaviours:
                j += 1
            
            maxlen = [(j - i + 1), i, j] if (j - i + 1) > maxlen[0] else maxlen
            i = j
        else:
            i += 1

    i, j = maxlen[1:]    
    x, y = x[i:j+1,:], y[i:j+1,:]

    HEAD, BASE_NECK, CENTER_SPINE, HINDPAW1, HINDPAW2, BASE_TAIL, MID_TAIL, TIP_TAIL = np.arange(8)
    points = [HEAD, BASE_NECK, CENTER_SPINE, HINDPAW1, HINDPAW2, BASE_TAIL, MID_TAIL, TIP_TAIL]
    points.remove(CENTER_SPINE)
    for p in points:
        x[:,p] = x[:,p] - x[:,CENTER_SPINE]
        y[:,p] = y[:,p] - y[:,CENTER_SPINE]
    x, y = x[:,points], y[:,points]
    t = np.repeat(np.reshape(np.arange(x.shape[0]), (1, x.shape[0])), x.shape[1], 0).T
    plot_len = round(j - i + 1 * 1000 / FPS, 1)
    
    for data in [x, y]:
        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot(111)
        cmap = mpl.cm.get_cmap('tab20')
        c = [cmap(n) for n in range(data.shape[1])]

        ax.plot(t, data, c)
        ax.axis('off')
        ax.text(0.8, 0.1, f"{plot_len} seconds", fontsize=6, transform=plt.gcf().transFigure, zorder=1000)
        plt.show()
    
    return maxlen

if __name__ == '__main__':
    # with open(stats_file, 'rb') as f:
    #     behaviour_stats = joblib.load(f)
    # behaviour_stats = all_behaviour_info_for_all_strains(label_info_file)
    # with open(stats_file, 'wb') as f:
    #     joblib.dump(behaviour_stats, f)   

    # preliminary_metrics(behaviour_stats)
    # samples_explained_by_behaviours(label_info_file)

    with open("./keypoint_data.pkl", "rb") as f:
        data = joblib.load(f)

    trim = [194000, 206000]

    groom = plot_keypoint_data(data, behaviours=BEHAVIOUR_LABELS["Groom"])

    behaviours = BEHAVIOUR_LABELS["Run"]
    behaviours.extend(BEHAVIOUR_LABELS["Walk"])
    behaviours.extend(BEHAVIOUR_LABELS["CW-Turn"])
    behaviours.extend(BEHAVIOUR_LABELS["CCW-Turn"])
    behaviours.extend(BEHAVIOUR_LABELS["Rear"])
    locomote = plot_keypoint_data(data, behaviours, limits=trim)

    ethogram_plot(data["labels"], [groom[1:], locomote[1:]], trim)

