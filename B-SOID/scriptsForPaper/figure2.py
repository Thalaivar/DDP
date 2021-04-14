import os
import cv2
import joblib
import ftplib
import numpy as np
import pandas as pd

# base_dir = "D:/IIT/DDP"
base_dir = "/Users/dhruvlaad/IIT/DDP"
SAVE_DIR = os.path.join(base_dir, "data/paper/figure2")
try: os.mkdir(SAVE_DIR)
except FileExistsError: pass

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage

import sys
sys.path.insert(0, os.path.join(base_dir, "DDP/B-SOID"))
from analysis import *
from BSOID.utils import get_video_and_keypoint_data
from tqdm import tqdm

def plot_behavioral_metrics(stats_file: str):
    save_dir = os.path.join(SAVE_DIR, "behavioural_metrics")
    try: os.mkdir(save_dir)
    except FileExistsError: pass

    grouped_stats = group_stats(stats_file)

    metrics = (
        "Total Duration",
        "Average Bout Length",
        "No. of Bouts"
    )

    labels = (
        "Total Duration (min)",
        "Average Bout Length (s)",
        "No. of Bouts"
    )

    save_names = ("TD", "ABL", "NB")

    ylimits = {
        'Groom'    : {'Total Duration': [0, 75], 'Average Bout Length': None, 'No. of Bouts': None},
        'Run'      : {'Total Duration': None, 'Average Bout Length': None, 'No. of Bouts': None},
        'Walk'     : {'Total Duration': None, 'Average Bout Length': [0, 1.5], 'No. of Bouts': None},
        'CW-Turn'  : {'Total Duration': None, 'Average Bout Length': None, 'No. of Bouts': None},
        'CCW-Turn' : {'Total Duration': None, 'Average Bout Length': None, 'No. of Bouts': None},
        'Point'    : {'Total Duration': [0, 20], 'Average Bout Length': [0, 20], 'No. of Bouts': [0, 1200]},
        'Rear'     : {'Total Duration': None, 'Average Bout Length': [0, 2], 'No. of Bouts': None},
        'N/A'      : {'Total Duration': None, 'Average Bout Length': None, 'No. of Bouts': None}    
    }

    for behaviour in BEHAVIOUR_LABELS.keys():
        print(f"Plotting for behaviour: {behaviour}")
        
        df = grouped_stats[behaviour].copy()
        gdfs = df.groupby("Strain")
        df["Total Duration"] /= 60

        df.loc[df.index[df["Strain"].str.contains("BTBR")], "Strain"] = r"BTBR T$^+$ ltpr3$^{tf}$/J"

        fig, axs = plt.subplots(nrows=len(metrics))
        for i, met in enumerate(metrics):
            metric_data_mean = gdfs[met].mean().sort_values()
            order = list(metric_data_mean.index)

            sns.stripplot(x='Strain', y=met, data=df, hue='Sex', jitter=False, order=order, ax=axs[i])
            axs[i].tick_params(grid_color='gray', grid_alpha=0.3, labelrotation=90, labelsize=8)
            axs[i].grid(True)
            axs[i].set_ylabel(labels[i], fontsize=12)
            axs[i].set_xlabel(None)
            axs[i].tick_params(axis='y', labelrotation=0, labelsize=8)
            if ylimits[behaviour][met] is not None:
                axs[i].set_ylim(ylimits[behaviour][met])

            rect_width = 0.5
            metric_data_std = gdfs[met].std().loc[order]
            for j in range(metric_data_mean.shape[0]):
                rect_height, mean = metric_data_std[j], metric_data_mean[j]
                rect_bottom_left = (j - (rect_width / 2), max(mean - rect_height, 0))
                if mean - rect_height < 0:
                    rect_size = (rect_width, mean)
                else:
                    rect_size = (rect_width, rect_height)
                axs[i].add_patch(plt.Rectangle(rect_bottom_left, *rect_size, edgecolor='k', linewidth=1.2, fill=False, zorder=1000))

                rect_bottom_left = (j - (rect_width / 2), mean)
                rect_size = (rect_width, rect_height)
                axs[i].add_patch(plt.Rectangle(rect_bottom_left, *rect_size, edgecolor='k', linewidth=1.2, fill=False, zorder=1000))

            axs[i].figure.set_size_inches(10, 1)
            axs[i].yaxis.set_major_locator(plt.MaxNLocator(5))

            if i > 0:
                axs[i].get_legend().set_visible(False)

        sns.despine(trim=True)
        axs[0].legend(loc='upper left', title='Sex', borderpad=0.5)
        fig.set_size_inches(10, 10)
        plt.subplots_adjust(hspace=0.6)
        if '/' in behaviour:
            behaviour = behaviour.replace('/', '-')
        plt.savefig(os.path.join(save_dir, f"{behaviour}_plot.jpg"), dpi=400, pad_inches=0, bbox_inches="tight")

def behaviour_usage_plot(label_info_file: str):
    save_dir = os.path.join(SAVE_DIR, "behaviour_usage_figure")
    try: os.mkdir(save_dir)
    except FileExistsError: pass

    prop = behaviour_usage_across_strains(label_info_file)
    prop.loc[prop.index[prop["Strain"].str.contains("BTBR")], "Strain"] = r"BTBR T$^+$ ltpr3$^{tf}$/J"
    prop = prop.set_index("Strain")
    prop_mean = prop.mean(axis=0).sort_values(ascending=False)
    prop_std = prop.std(axis=0).loc[list(prop_mean.index)]

    fig, ax = plt.subplots(figsize=(9,6))
    ax.errorbar(
        x=list(prop_mean.index), 
        y=prop_mean.values, 
        yerr=prop_std.values, 
        fmt="-k", 
        ecolor=(0.59, 0.59, 0.61, 1.0), 
        elinewidth=1, 
        capsize=4,
        linewidth=3
    )
    
    ax.set_xlabel('Identified Phenotypes', fontsize=18)
    ax.set_ylabel('Proportion Usage', fontsize=18)
    sns.despine(trim=True)
    ax.tick_params(axis='x', labelrotation=90, labelsize=14)
    plt.savefig(os.path.join(save_dir, "behaviour_usage_plot.jpg"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    usage = prop.groupby("Strain").sum()
    usage = usage.div(usage.sum(axis=1), axis=0).T.clip(lower=0.01)

    ax = sns.heatmap(usage, xticklabels=True, yticklabels=True, cbar_kws={"pad": 0.01}, cmap="PuBuGn")
    plt.gcf().set_size_inches(15.5, 4)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.savefig(os.path.join(save_dir, "strainwise_usage_plot.jpg"), dpi=400, bbox_inches='tight', pad_inches=0)
    plt.show() 

def download_video_file(label_info_file: str):
    save_dir = os.path.join(SAVE_DIR, "vignette_figure")
    try: os.mkdir(save_dir)
    except FileExistsError: pass

    with open(label_info_file, "rb") as f:
        info = joblib.load(f)
    N = len(info["Labels"])
    idx = np.random.randint(0, N, 1)[0]

    metadata = {key: val[idx] for key, val in info.items()}

    session = ftplib.FTP("ftp.box.com")
    session.login("ae16b011@smail.iitm.ac.in", "rSNxWCBv1407")
    data_fname, vid_fname = get_video_and_keypoint_data(session, metadata, save_dir)

    from BSOID.data import extract_to_csv
    data_fname = extract_to_csv(os.path.join(save_dir, data_fname), save_dir)

    import pandas as pd
    from BSOID.preprocessing import likelihood_filter
    from BSOID.bsoid import FPS
    data = pd.read_csv(data_fname)
    fdata, perc_filt = likelihood_filter(data, fps=FPS, end_trim=0, clip_window=0)
    
    shape = fdata['x'].shape
    print(f'Preprocessed {shape} data, with {round(perc_filt, 2)}% data filtered')
    
    metadata["data_filename"] = data_filename
    metadata["vid_filename"] = vid_filename

    with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
        joblib.dump(metadata, f)

def save_frames_and_loc_data(behaviour, n=10):
    save_dir = os.path.join(SAVE_DIR, "vignette_figure", f"{behaviour}_clips")
    with open(os.path.join(save_dir, "metadata.pkl"), "rb") as f:
        metadata = joblib.load(f)
    
    video_file = os.path.join(save_dir, metadata["vid_filename"])
    keypoint_file = os.path.join(save_dir, metadata["data_filename"])
    labels = metadata["Labels"]

    i, locs = 0, []

    while i < len(labels):
        if labels[i] in BEHAVIOUR_LABELS[behaviour]:
            j = i + 1
            while j < len(labels) and (labels[j] in BEHAVIOUR_LABELS[behaviour]):
                j += 1
            
            if (j - i + 1) > MIN_BOUT_LENS[lab]:
                locs.append([i, j, (j + i - 1)])

            i = j
        else:
            i += 1
    
    locs = sorted(locs, key=lambda x: x[-1])[-n:]
    
    count, i = 0, 0
    video = cv2.VideoCapture(video_file)
    success, image = video.read()
    while success and locs:
        if i == locs[0][0]:
            clip_dir = os.path.join(save_dir, f"clips_{count}")
            while success and i <= locs[0][1]:
                cv2.imwrite(os.path.join(clip_dir, f"{behaviour}_frame_{i}.jpg"), image)
                success, image = video.read()
                i += 1

            x, y = 
            del locs[0]
        else:
            i += 1
            success, image = video.read()

def make_vignettes(lab, idxs, weights):
    assert len(idxs) == len(weights)
    assert sum(weights) == 1

    save_dir = os.path.join(SAVE_DIR, "vignette_figure")

    frames = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.split('_')[0] == lab]
    frames.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    if idxs is not None:
        frames = [cv2.imread(frames[i]) for i in idxs]

    img = sum(f * w for f, w in zip(frames, weights))

    save_dir = os.path.join(save_dir, "vignettes")
    try: os.mkdir(save_dir)
    except FileExistsError: pass
    cv2.imwrite(os.path.join(save_dir, f"{lab}_vignette.jpg"), img)

if __name__ == "__main__":
    # plot_behavioral_metrics(stats_file="../../data/analysis/stats.pkl")
    behaviour_usage_plot(label_info_file="../../data/analysis/label_info.pkl")
    
    # download_video_file("../../data/analysis/label_info.pkl")
    # save_frames("Rear")

    # make_vignettes("Rear", [16, 22, 24], [1/3, 1/3, 1/3])
