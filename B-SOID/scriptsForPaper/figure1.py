import os
import cv2
import joblib
import numpy as np
import pandas as pd

SAVE_DIR = "D:/IIT/DDP/data/paper/figure1"
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
from BSOID.utils import get_random_video_and_keypoints
from tqdm import tqdm

def download_video_and_keypt_data():
    data_fname, _ = get_random_video_and_keypoints("D:/IIT/DDP/data/paper/MergedMetaList_2019-04-18_strain-survey-mf-subset.csv", SAVE_DIR)

    from BSOID.data import extract_to_csv
    extract_to_csv(os.path.join(SAVE_DIR, data_fname), "D:/IIT/DDP/data/paper")

    import pandas as pd
    from BSOID.preprocessing import likelihood_filter
    from BSOID.bsoid import FPS
    data = pd.read_csv([os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".csv")][0])
    fdata, perc_filt = likelihood_filter(data, fps=FPS, conf_threshold=0.3, end_trim=0, clip_window=0)

    shape = fdata['x'].shape
    print(f'Preprocessed {shape} data, with {round(perc_filt, 2)}% data filtered')

    with open(os.path.join(SAVE_DIR, "rawdata.pkl"), "wb") as f:
        joblib.dump(fdata, f)

def get_frames_from_video(video_file, keypoints_file, n=10):
    save_dir = os.path.join(SAVE_DIR, "frame_keypoint_fig")
    try: os.mkdir(save_dir)
    except FileExistsError: pass
    
    with open(keypoints_file, "rb") as f:
        keypoints = joblib.load(f)
    x, y = keypoints['x'], keypoints['y']

    assert x.shape[0] == y.shape[0]
    N = x.shape[0]

    idxs = np.random.randint(0, N, n)
    idxs.sort()
    x, y = x[idxs], y[idxs]

    count, frames = 0, []
    video = cv2.VideoCapture(video_file)
    success, image = video.read()
    while success:
        if count in idxs:
            frames.append(image)
        count += 1

        success, image = video.read()
    
    assert count == N, "# of frames in video does not match with # of keypoint-datapoints"

    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(save_dir, f"frame{i}.jpg"), f)
    
    with open(os.path.join(save_dir, "keypointdata.pkl"), "wb") as f:
        joblib.dump([x, y], f)

def keypoint_plot(image_file, keypoint_data_file, idx, deg=0):
    save_dir = os.path.join(SAVE_DIR, "frame_keypoint_fig")
    with open(keypoint_data_file, "rb") as f:
        x, y = joblib.load(f)
    
    x, y = x[idx], y[idx]

    HEAD, BASE_NECK, CENTER_SPINE, HINDPAW1, HINDPAW2, BASE_TAIL, MID_TAIL, TIP_TAIL = np.arange(8)
    link_connections = ([BASE_TAIL, CENTER_SPINE],
                        [CENTER_SPINE, BASE_NECK],
                        [BASE_NECK, HEAD],
                        [BASE_TAIL, HINDPAW1], [BASE_TAIL, HINDPAW2],
                        [BASE_TAIL, MID_TAIL],
                        [BASE_TAIL, TIP_TAIL])
    
    fig, ax = plt.subplots()
    img = ndimage.rotate(plt.imread(image_file), deg)
    ax.imshow(img, extent=[0, img.shape[0], 0, img.shape[1]])
    
    for link in link_connections:
        h, t = link
        plt.plot([x[h], x[t]], [y[h], y[t]], linewidth=2, color="y")
    
    cmap = mpl.cm.get_cmap("tab20")
    for idx in np.arange(8):
        plt.scatter(x[idx], y[idx], color=cmap(idx), s=30)
    
    plt.xlim([180, 320])
    plt.ylim([100, 300])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()
    
    plt.savefig(os.path.join(save_dir, "keypoint_plot.jpg"), bbox_inches="tight", pad_inches=0)
    fig.show()

    fig, ax = plt.subplots()
    img = ndimage.rotate(plt.imread(image_file), deg)
    ax.imshow(img, extent=[0, img.shape[0], 0, img.shape[1]])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()
    plt.savefig(os.path.join(save_dir, "original_image.jpg"), bbox_inches="tight", pad_inches=0)

def get_autocorrelation_arrays():
    save_dir = os.path.join(SAVE_DIR, "autocorrelation_fig")
    autocorr = autocorrelation(tmax=3000, n=1000)

    block_lens = [100, 250, 500, 1000, 1500]
    block_autocorr = block_shuffle_autocorrelation(block_lens, tmax=3000, n=1000)

    np.save(arr=autocorr, file=os.path.join(save_dir, "autocorr.npy"))
    with open(os.path.join(save_dir, "block_autocorr.pkl"), "wb") as f:
        joblib.dump([block_lens, block_autocorr], f)

def autocorrelation_plots(autocorr, block_autocorr, block_lens, fps):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    n, m = autocorr.shape
    x = np.arange(m) * 1000 / 30

    for i in range(n):
        ax[0].plot(x, autocorr[i], color='gray', alpha=0.005, linewidth=1.5)
    
    ax[0].plot(x, autocorr.mean(axis=0), color='black', linewidth=2)

    x = np.arange(block_autocorr.shape[1]) * 1000 / 30
    cmap = mpl.cm.get_cmap("Blues")
    for i in range(block_autocorr.shape[0]):
        ax[1].plot(x, block_autocorr[i], color=cmap(100 * (i+1)), linewidth=2, label=f"{block_lens[i]} ms")
    
    for ax_ in ax:
        ax_.set_xlabel("Time Lag (ms)", fontsize=14)
        ax_.set_ylabel("Autocorrelation", fontsize=14)
        ax_.set_xticks([0, 1000, 2000, 3000])

    leg = ax[1].legend(loc="upper right", frameon=False, prop={'size': 12})
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "autocorrelation_fig", "autocorr.jpg"), bbox_inches="tight", pad_inches=0)

def plot_centroid_velocity(tspan, fps):
    save_dir = os.path.join(SAVE_DIR, "centroid_velocity_fig")
    
    with open(os.path.join(save_dir, "metadata.pkl"), "rb") as f:
        metadata = joblib.load(f)
    
    from BSOID.preprocessing import smoothen_data
    centroid_idx = 2
    x, y = metadata["keypoints"]['x'][:,centroid_idx], metadata["keypoints"]['y'][:,centroid_idx]
    x, y = smoothen_data(x), smoothen_data(y)
    x, y = (x[1:] - x[0:-1]) * 0.0264 / (1 / fps), (y[1:] - y[0:-1]) * 0.0264 / (1 / fps)
    x, y = x/100, y/100
    vel = np.sqrt(x ** 2 + y ** 2)

    start_idx, tspan = 14500, tspan * fps
    vel = vel[start_idx:start_idx+tspan]

    fig, ax = plt.subplots(figsize=(12,1.3))
    plt.plot(np.arange(vel.shape[0]), vel, color="k", linewidth=1)
    plt.ylim([0, 0.11])
    plt.xlim([0,vel.shape[0]])
    ax.set_xticks([tspan])
    ax.set_yticks([0, 0.11])
    ax.set_xticklabels([f"{int(tspan / fps)} seconds"], fontsize=10)
    ax.set_yticklabels(["0", f"{0.1} m/s"])
    ax.yaxis.tick_right()
    ax.tick_params(axis=u'both', which=u'both',length=0)
    for _, spine in ax.spines.items():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "centroid_velocity.jpg"), bbox_inches="tight", pad_inches=0.5)
    plt.show()

def idx2group_map():
    i, idx2grp = 0, {}
    for lab, idxs in BEHAVIOUR_LABELS.items():
        for idx in idxs:
            idx2grp[idx] = (i, lab)
        i += 1
    return idx2grp

def ethogram_plot(tspan, fps):
    cmap = mpl.cm.get_cmap('tab20')
    idx2grp = idx2group_map()

    start_idx, tspan = 14500, tspan * fps
    with open(os.path.join(SAVE_DIR, "centroid_velocity_fig", "metadata.pkl"), "rb") as f:
        metadata = joblib.load(f)
    
    all_labels = metadata["labels"][start_idx:start_idx+tspan]
    height, N = 1, len(all_labels)
    labels = [idx2grp[l][0] for l in all_labels]

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 3), gridspec_kw={'height_ratios': [1, 2.75]})
    i, pat = 0, []
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[i] == labels[j]:
            j += 1
        pat.append(patches.Rectangle((i, 0), (j - i + 1), height, color=cmap(labels[i])))
        i = j
    
    [ax[0].add_patch(p) for p in pat]
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    for _, spine in ax[0].spines.items():
        spine.set_visible(False)
    ax[0].set_xlim([0, N])
    ax[0].set_ylabel("Ethogram", fontsize=10)

    height = 0.5
    k, ylocs = 0, {}
    for _, cl in idx2grp.items():
        idx, lab = cl
        if idx not in ylocs:
            ylocs[idx] = (k*(height + 0.1), lab)
            k += 1
    
    i, pat = 0, []
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[i] == labels[j]:
            j += 1
        y = ylocs[labels[i]][0]
        pat.append(patches.Rectangle((i, y), (j - i + 1), height, color=cmap(labels[i])))
        i = j    

    [ax[1].add_patch(p) for p in pat]
    ax[1].set_xlim([0, N])
    ax[1].set_ylim([0 - 0.1, (k+1)*(height+0.1) - 0.1])

    for _, lab in ylocs.items():
        y = lab[0] + (height / 2)
        ax[1].plot(np.arange(N), y * np.ones((N,)), color='gray', linestyle='-', linewidth=0.77, alpha=0.1)
    ax[1].set_yticks([lab[0] + (height / 2) for _, lab in ylocs.items()])
    ax[1].set_yticklabels([lab[1] for _, lab in ylocs.items()], fontsize=9)
    ax[1].yaxis.tick_right()
    ax[1].set_ylabel("Phenotypes", fontsize=10)
    ax[1].set_xticks([tspan])
    ax[1].set_xticklabels([f"{int(tspan / fps)} seconds"], fontsize=10)
    ax[1].tick_params(axis=u'both', which=u'both',length=0)
    for _, spine in ax[1].spines.items():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ethogram_fig", "ethogram_plot.jpg"), bbox_inches="tight", pad_inches=0, dpi=600)
    plt.show()

def pc_space_plot():
    bsoid = BSOID.load_config("D:/IIT/DDP/data", "dis")
    feats = bsoid.load_features()


if __name__ == "__main__":
    # download_video_and_keypt_data()
    # vid_file = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".avi")][0]
    # keypt_file = os.path.join(SAVE_DIR, "rawdata.pkl")
    # get_frames_from_video(vid_file, keypt_file)
    
    # idx = 0
    # image_file = f"../../data/paper/figure1/frame_keypoint_fig/frame{idx}.jpg"
    # keypoint_data = "../../data/paper/figure1/frame_keypoint_fig/keypointdata.pkl"
    # keypoint_plot(image_file, keypoint_data, idx, deg=90)

    from BSOID.bsoid import BSOID
    bsoid = BSOID.load_config("D:/IIT/DDP/data", "dis")

    # autocorr = np.load(os.path.join(SAVE_DIR, "autocorrelation_fig", "autocorr.npy"))
    # with open(os.path.join(SAVE_DIR, "autocorrelation_fig", "block_autocorr.pkl"), "rb") as f:
    #     block_lens, block_autocorr = joblib.load(f)
    # autocorrelation_plots(autocorr, block_autocorr, block_lens, bsoid.fps)

    # metadata = pd.read_csv("../../data/analysis/StrainSurveyMetaList_2019-04-09.tsv", sep='\t')
    # metadata = metadata[metadata["Strain"] == "C57BL/6J"]
    # metadata = dict(metadata.iloc[np.random.randint(0, metadata.shape[0], 1)[0]])
    # label_info_file = "../../data/analysis/label_info.pkl"
    # data_dir = "/projects/kumar-lab/StrainSurveyPoses"
    # metadata = get_keypoint_data(metadata, label_info_file, data_dir)
    # with open(os.path.join(SAVE_DIR, "centroid_velocity_fig", "metadata.pkl"), "wb") as f:
    #     joblib.dump(metadata, f)
    
    plot_centroid_velocity(200, bsoid.fps)
    # ethogram_plot(200, bsoid.fps)