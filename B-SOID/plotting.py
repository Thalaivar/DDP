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
plot_dir = './plots/'

# with open(stats_file, 'rb') as f:
#     behaviour_stats = joblib.load(f)
# behaviour_stats = all_behaviour_info_for_all_strains(label_info_file)
# with open(stats_file, 'wb') as f:
#     joblib.dump(behaviour_stats, f)   

def idx2group_map():
    i, idx2grp = 0, {}
    for lab, idxs in BEHAVIOUR_LABELS.items():
        for idx in idxs:
            idx2grp[idx] = i
        i += 1
    return idx2grp

#################################################################################################
#                                       Behaviour Metrics                                       #
#################################################################################################
def sort_info_df(info, metric):
    N = info.shape[0]
    strains = {}
    for strain in info['Strain']:
        strains[strain] = []
        
    for i in range(N):
        data = info.iloc[i]
        strains[data['Strain']].append(data[metric])
    
    for key, value in strains.items():
        strains[key] = sum(value)/len(value)
    
    strains = dict(sorted(strains.items(), key=lambda item: item[1]))
    return strains.keys()

def plot_behaviour_metric(info, metric, order, ylabel=None):        
    ylabel = metric if ylabel is None else ylabel
    fig = plt.figure()
    g = sns.catplot(x='Strain', y=metric, data=info, hue='Sex', jitter=False, legend=False, order=order, ci=0.95)
    ax = g.axes[0,0]
    ax.tick_params(grid_color='gray', grid_alpha=0.3, labelrotation=90, labelsize=8)
    ax.grid(True)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(None, fontsize=12)
    ax.legend(loc='upper left', title='Sex', borderpad=0.5)
    plt.gcf().set_size_inches(10, 2.2)

    for idx, strain in enumerate(order):
        rect_width = 0.5
        strain_data = info.loc[info['Strain'] == strain, metric]
        rect_height, mean = strain_data.std(), strain_data.mean()
        rect_bottom_left = (idx - (rect_width / 2), max(mean - rect_height, 0))
        if mean - rect_height < 0:
            rect_size = (rect_width, mean)
        else:
            rect_size = (rect_width, rect_height)
        ax.add_patch(plt.Rectangle(rect_bottom_left, *rect_size, edgecolor='k', linewidth=1.2, fill=False, zorder=1000))

        rect_bottom_left = (idx - (rect_width / 2), mean)
        rect_size = (rect_width, rect_height)
        ax.add_patch(plt.Rectangle(rect_bottom_left, *rect_size, edgecolor='k', linewidth=1.2, fill=False, zorder=1000))

    return fig, ax

def plot_all_metrics_for_behaviour(behaviour_stats, behaviour, ylimits):
    stats = behaviour_stats[behaviour]
    info = pd.DataFrame.from_dict(stats)
    info['Total Duration'] = info['Total Duration']/60

    metrics = ['Total Duration', 'Average Bout Length', 'No. of Bouts']
    labels = ['Total Duration (min)', 'Average Bout Length (s)', 'No. of Bouts']
    save_names  = ['TD', 'ABL', 'NB']

    figs = []
    for i in range(len(metrics)):
        strains = list(sort_info_df(info, metrics[i]))
        fig, ax = plot_behaviour_metric(info, metrics[i], strains, labels[i])
        
        ax.tick_params(axis='y', labelrotation=0, labelsize=8)
        if ylimits[metrics[i]] is not None:
            ax.set_ylim(ylimits[metrics[i]])
        behaviour = behaviour.replace('/', '-')
        sns.despine(trim=True)
        plt.savefig(f'{plot_dir}/{save_names[i]}_{behaviour}', dpi=500, bbox_inches='tight')
        figs.append(fig)
    
    return figs


def preliminary_metrics(behaviour_stats):
    print('Calculating preliminary metrics for behaviours:')
    ylimits = {
        'Groom'    : {'Total Duration': None, 'Average Bout Length': None, 'No. of Bouts': None},
        'Run'      : {'Total Duration': None, 'Average Bout Length': None, 'No. of Bouts': None},
        'Walk'     : {'Total Duration': None, 'Average Bout Length': [0, 1.5], 'No. of Bouts': None},
        'CW-Turn'  : {'Total Duration': None, 'Average Bout Length': None, 'No. of Bouts': None},
        'CCW-Turn' : {'Total Duration': None, 'Average Bout Length': None, 'No. of Bouts': None},
        'Point'    : {'Total Duration': [0, 20], 'Average Bout Length': [0, 20], 'No. of Bouts': [0, 1200]},
        'Rear'     : {'Total Duration': None, 'Average Bout Length': [0, 2], 'No. of Bouts': None},
        'N/A'      : {'Total Duration': None, 'Average Bout Length': None, 'No. of Bouts': None}    
    }

    grouped_stats = group_stats(behaviour_stats)
    for behaviour in BEHAVIOUR_LABELS.keys():
        print(f'Making plots for {behaviour}...')
        plot_all_metrics_for_behaviour(grouped_stats, behaviour, ylimits[behaviour])

########################################################################################################################
#                                               Behaviour Usage                                                        #
########################################################################################################################
def sort_usage_info(info):
    N = info.shape[0]
    behaviours = {}
    for behaviour in info['Behaviour']:
        behaviours[behaviour] = []
        
    for i in range(N):
        data = info.iloc[i]
        behaviours[data['Behaviour']].append(data['Usage'])
    
    for key, value in behaviours.items():
        value = np.array(value)
        behaviours[key] = [value.mean(), value.std()]
    
    behaviours = dict(sorted(behaviours.items(), key=lambda item: item[1][0], reverse=True))
    return behaviours

def samples_explained_by_behaviours(label_info_file):
    print('Calculating proportion of samples explained by the behaviours:')
    info = calculate_behaviour_usage(label_info_file)
    info = sort_usage_info(info)
    
    x, mean_usage, std_usage = [], [], []
    for key, val in info.items():
        x.append(key)
        mean_usage.append(val[0])
        std_usage.append(val[1])
    
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.errorbar(x=x, y=mean_usage, yerr=std_usage, fmt='-k', ecolor=(0.59, 0.59, 0.61, 0.7), elinewidth=5, linewidth=3)
    ax.set_xlabel('Sorted Behaviours')
    ax.set_ylabel('Proportion Usage')
    sns.despine(trim=True)
    ax.tick_params(axis='x', labelrotation=90)
    plt.savefig(f'{plot_dir}/prop.png', dpi=600, bbox_inches='tight')
    fig.show()

    usage_data = behaviour_usage_across_strains(label_info_file)
    usage_datamat = usage_data.pivot_table(index='Behaviour', columns='Strain', values='Usage')

    ax = sns.heatmap(usage_datamat, xticklabels=True, yticklabels=True, cmap='mako')
    plt.gcf().set_size_inches(15.5, 4)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.savefig(f'{plot_dir}/usage.png', dpi=600, bbox_inches='tight')
    plt.show() 

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

def plot_keypoints(keypoint_file, clf, behaviour):
    with open(keypoint_file, "rb") as f:
        fdata = joblib.load(keypoint_file)
    
    feats = frameshift_features(fdata, STRIDE_WINDOW, FPS, extract_feats, window_extracted_feats)
    labels = frameshift_predict(feats, clf, STRIDE_WINDOW)
    # x = windowed_feats(fdata["x"], STRIDE_WINDOW, mode="mean")
    # y = windowed_feats(fdata["y"], STRIDE_WINDOW, mode="mean")
    x, y = fdata['x'], fdata['y']
    if abs(len(labels) - x.shape[0]) <= STRIDE_WINDOW:
        x, y = x[:len(labels), :], y[:len(labels), :]
    else:
        raise ValueError(f"mismatch in no. of labels ({len(labels)}) and no. of datapoints ({x.shape[0]})")
    HEAD, BASE_NECK, CENTER_SPINE, HINDPAW1, HINDPAW2, BASE_TAIL, MID_TAIL, TIP_TAIL = np.arange(8)
    points = [HEAD, BASE_NECK, CENTER_SPINE, HINDPAW1, HINDPAW2, BASE_TAIL, MID_TAIL, TIP_TAIL]
    points.remove(CENTER_SPINE)
    for p in points:
        x[:,p] = x[:,p] - x[:,CENTER_SPINE]
        y[:,p] = y[:,p] - y[:,CENTER_SPINE]

    # find longest bout for given behaviour
    idx2group, i, maxlen = idx2group_map(), 0, [0, None, None]
    while i < len(labels):
        if idx2group[labels[i]] == idx2group[behaviour]:
            j = i + 1
            while j < len(labels) and labels[j] == labels[i]:
                j += 1
            
            maxlen = [(j - i + 1), i, j] if (j - i + 1) > maxlen[0] else maxlen
        
            i = j
        else:
            i += 1
    
    i, j = maxlen[1:]
    print(maxlen)
    t = np.repeat(np.reshape(np.arange((j-i+1)), (1, j-i+1)), x.shape[1], 0)
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    cmap = mpl.cm.get_cmap('tab20')
    c = [cmap(x) for x in range(x.shape[1])]
    ax[0].plot(t, x[i:j+1,:].T, c)
    ax[1].plot(t, y[i:j+1,:].T, c)
    plt.show()

def ethogram_plot(label_info_file, metadata=None):
    with open(label_info_file, 'rb') as f:
        info = joblib.load(f)
    
    N = len(info['Strain'])
    if metadata is not None:
        for i in range(N):
            if (
                info['Strain'][i] == metadata['Strain'] and
                info['MouseID'][i] == metadata['MouseID'] and
                info['Sex'][i] == metadata['Sex'] and
                info['NetworkFilename'][i] == metadata['NetworkFilename']
            ):
                labels = info['Labels'][i]
                break
    else:
        labels = info['Labels'][2494]

    cmap = mpl.cm.get_cmap('tab20')
    height, dt, N = 1, 1 / FPS, len(labels)
    x = [i * dt / 60 for i in range(len(labels))]
    idx2grp = idx2group_map()
    labels = [idx2grp[l] for l in labels]

    fig = plt.figure(figsize=(12, 0.5))
    ax = fig.add_subplot(111)
    pat = [patches.Rectangle((x[i], 0), dt, height, color=cmap(labels[i])) for i in range(len(labels))]
    [ax.add_patch(pat[i]) for i in tqdm(range(len(labels)))]
    plt.xlim([0, x[-1] + dt])
    plt.ylim([0, 1])
    plt.xticks([i * 10 for i in range(round(x[-1] / 10))])
    plt.yticks([])
    plt.xlabel('Mins')
    sns.despine()
    plt.savefig(f'{plot_dir}/ethogram.png', dpi=600, bbox_inches='tight')

if __name__ == '__main__':
    # with open(stats_file, 'rb') as f:
    #     behaviour_stats = joblib.load(f)
    # behaviour_stats = all_behaviour_info_for_all_strains(label_info_file)
    # with open(stats_file, 'wb') as f:
    #     joblib.dump(behaviour_stats, f)   

    # preliminary_metrics(behaviour_stats)
    # samples_explained_by_behaviours(label_info_file)

    # label_info_file = 'D:/IIT/DDP/data/analysis/label_info.pkl'
    # metadata = {
    #     'NetworkFilename': 'LL6-B2B/2017-11-25_SPD/B6J_Male_S6938572M-3-PSY.avi',
    #     'Strain': 'C57BL/6J',
    #     'MouseID': 'B6J_Male_S6938572M-3-PSY',
    #     'Sex': 'M'
    # }
    # ethogram_plot(label_info_file, metadata)

    clf_file = "/Users/dhruvlaad/IIT/DDP/data/output/dis_classifiers.sav"
    with open(clf_file, "rb") as f:
        clf = joblib.load(f)
    
    keypoint_file = "/Users/dhruvlaad/IIT/DDP/data/analysis/NZW#LacJ-M-LL4-1_001058-M-MP16-8-42444-5-S193.pkl"
    plot_keypoints(keypoint_file, clf, BEHAVIOUR_LABELS["Run"][0])