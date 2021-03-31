import os
import cv2
import joblib
import numpy as np

SAVE_DIR = "D:/IIT/DDP/data/paper/figure1"
try: os.mkdir(SAVE_DIR)
except FileExistsError: pass

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage

def download_video_and_keypt_data():
    import sys
    sys.path.insert(0, "D:/IIT/DDP/DDP/B-SOID")
    from BSOID.utils import get_random_video_and_keypoints
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

if __name__ == "__main__":
    # download_video_and_keypt_data()
    # vid_file = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".avi")][0]
    # keypt_file = os.path.join(SAVE_DIR, "rawdata.pkl")
    # get_frames_from_video(vid_file, keypt_file)
    idx = 0
    image_file = f"../../data/paper/figure1/frame_keypoint_fig/frame{idx}.jpg"
    keypoint_data = "../../data/paper/figure1/frame_keypoint_fig/keypointdata.pkl"
    keypoint_plot(image_file, keypoint_data, idx, deg=90)