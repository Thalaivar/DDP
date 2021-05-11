import os
import cv2
import h5py
import logging

import numpy as np
import pandas as pd
from BSOID.bsoid import BSOID
from itertools import combinations
from BSOID.utils import alphanum_key
from BSOID.features import extract_comb_feats as extract_feats
from BSOID.data import process_h5py_data, bsoid_format
from BSOID.preprocessing import smoothen_data, likelihood_filter
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def extract_data_from_video(bsoid: BSOID, raw_data_file: str, video_file: str, extract_frames=False):
    video_name = os.path.split(video_file)[-1][:-4]
    frame_dir = os.path.join(os.path.split(video_file)[0], f"{video_name}_frames")

    try: 
        os.mkdir(frame_dir)
        extract_frames = True
    except FileExistsError: 
        pass

    if extract_frames:
        logger.info(f"extracting frames from {video_file} to {frame_dir}")
        video = cv2.VideoCapture(video_file)
        
        video_fps = video.get(cv2.CAP_PROP_FPS)
        assert video_fps == bsoid.fps, f"video fps ({video_fps}) not matching ({bsoid.fps})"
        extract_frames_from_video(video, frame_dir, **bsoid.trim_params)
    
    # get keypoint data from video
    conf, pos = process_h5py_data(h5py.File(raw_data_file, "r"))
    bsoid_data = bsoid_format(conf, pos)
    fdata, perc_filt = likelihood_filter(bsoid_data, bsoid.fps, bsoid.conf_threshold, bsoid.bodyparts, **bsoid.trim_params)
    if perc_filt > bsoid.filter_thresh:
        logger.warning(f"% data filtered from {os.path.split(raw_data_file)[-1]} too high ({perc_filt}%)")

    frames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir)], key=alphanum_key)
    assert len(frames) == fdata['x'].shape[0], f"video:{video_name}: # of frames ({len(frames)}) does not match with no. of datapoints ({fdata['x'].shape[0]})"
    
    return fdata, frames

def extract_frames_from_video(video: cv2.VideoCapture, frame_dir: str, end_trim: int, clip_window: int):
    fps = video.get(cv2.CAP_PROP_FPS)
    
    hour_len = 55 * 60 * fps
    end_trim *= (fps * 60)
    clip_window *= (fps * 60)
    
    assert clip_window + 2 * end_trim < hour_len, "(end_trim + clip_window) is too large"

    success, image = video.read()
    frame_idx, count = 0, 0
    while success:
        if (count > 2 * end_trim) and (count <= 2 * end_trim + clip_window):
            cv2.imwrite(os.path.join(frame_dir, f"frame{frame_idx}.jpg"), image)
            frame_idx += 1
        success, image = video.read()
        count += 1
    
    logger.info(f"extracted {frame_idx + 1} frames at {fps} FPS")

def get_all_bouts(labels):
    labels = labels.astype("int")
    classes = np.unique(labels)
    vid_locs = [[] for i in range(np.max(classes)+1)]
    
    i = 0
    while i < len(labels)-1:
        curr_vid_locs = {'start': 0, 'end': 0}
        class_id = labels[i]
        curr_vid_locs['start'] = i
        while i < len(labels)-1 and labels[i] == labels[i+1]:
            i += 1
        curr_vid_locs['end'] = i
        vid_locs[class_id].append(curr_vid_locs)
        i += 1
    return vid_locs 

def example_video_segments(labels, bout_length, n_examples):
    class_vid_locs = get_all_bouts(labels)

    # filter out all videos smaller than len bout_length
    for k, class_vids in enumerate(class_vid_locs):
        for i, vid in enumerate(class_vids):
            if vid['end'] - vid['start'] < bout_length:
                del class_vid_locs[k][i]

    # get longest vids
    for k, class_vids in enumerate(class_vid_locs):
        class_vids.sort(key=lambda loc: loc['start']-loc['end'])
        if len(class_vids) > n_examples:
            class_vid_locs[k] = class_vids[0:n_examples]        

    return class_vid_locs

def labels_for_video(bsoid: BSOID, raw_data_file: str, video_file: str, extract_frames=False):
    filtered_data, frames = extract_data_from_video(bsoid, raw_data_file, video_file, extract_frames)
    x_raw, y_raw = filtered_data['x'], filtered_data['y']
    assert x_raw.shape == y_raw.shape
    _, n_dpoints = x_raw.shape

    win_len = np.int(np.round(0.05 / (1 / bsoid.fps)) * 2 - 1) if bsoid.stride_window is None else bsoid.stride_window // 2
    x, y = np.zeros_like(x_raw), np.zeros_like(y_raw)
    for i in range(n_dpoints):
        x[:,i] = smoothen_data(x_raw[:,i], win_len)
        y[:,i] = smoothen_data(y_raw[:,i], win_len)
    
    links = [np.array([x[:,i] - x[:,j], y[:,i] - y[:,j]]).T for i, j in combinations(range(n_dpoints), 2)]
    link_lens = np.vstack([np.linalg.norm(link, axis=1) for link in links]).T
    link_angles = np.vstack([np.arctan2(link[:,1], link[:,0]) for i, link in enumerate(links)]).T

    for i in range(link_lens.shape[1]):
        link_lens[:,i] = np.array(pd.Series(link_lens[:,i]).rolling(bsoid.stride_window, min_periods=1, center=True).mean())
    for i in range(link_angles.shape[1]):
        link_angles[:,i] = np.array(pd.Series(link_angles[:,i]).rolling(bsoid.stride_window, min_periods=1, center=True).sum())
    feats = np.hstack((link_angles, link_lens))
    clf = bsoid.load_classifier()
    labels = clf.predict(feats)
    return labels.reshape(1,-1)[0], frames

def frameshift_features(filtered_data, stride_window, fps, feats_extractor, windower):
    if not isinstance(filtered_data, list):
        filtered_data = [filtered_data]

    feats = [feats_extractor(data, fps, stride_window) for data in filtered_data]

    assert len(feats) == 1
    feats = feats[0]

    # frameshift and stack features into bins
    fs_feats = []
    for s in range(stride_window):
       fs_feats.append(feats[s:,:])

    fs_feats = windower(fs_feats, stride_window)

    return fs_feats

def frameshift_predict(feats, clf, stride_window):
    labels = []
    for f in feats:
        labels.append(clf.predict(f))

    # compatibility for CatBoostClassifier
    for i, lab in enumerate(labels):
        labels[i] = lab.reshape(1, -1)[0]

    for n in range(len(labels)):
        labels[n] = labels[n][::-1]
    
    labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
    
    for n, l in enumerate(labels):
        labels_pad[n][0:len(l)] = l
        labels_pad[n] = labels_pad[n][::-1]

        if n > 0:
            labels_pad[n][0:n] = labels_pad[n-1][0:n]
    
    fs_labels = labels_pad.astype(int)
    fs_labels2 = []
    for l in range(stride_window):
        fs_labels2.append(fs_labels[l])
    fs_labels = np.array(fs_labels2).flatten('F')

    return fs_labels

def videomaker(frames, fps, outfile):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    height, width, _ = frames[0].shape

    video = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
    for i in range(len(frames)):
        video.write(frames[i].astype(np.uint8))

    cv2.destroyAllWindows()
    video.release()


def create_class_examples(bsoid: BSOID, video_dir: str, min_bout_len: int, n_examples: int, outdir: str):
    clf = bsoid.load_classifier()
    min_bout_len = bsoid.fps * min_bout_len // 1000

    try: os.mkdir(outdir)
    except FileExistsError: pass

    video_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".avi")])
    raw_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".h5")])

    all_videos = []
    for raw_file, video_file in zip(raw_files, video_files):
        video_name = os.path.split(video_file)[-1][:-4]
        # get trimmed labels and frames
        # fdata, frames = extract_data_from_video(bsoid, raw_file, video_file)        
        # feats = frameshift_features(fdata, bsoid.stride_window, bsoid.fps, extract_feats, window_extracted_feats)
        # labels = frameshift_predict(feats, clf, bsoid.stride_window)
        labels, frames = labels_for_video(bsoid, raw_file, video_file)
        
        if labels.size != len(frames):
            if len(frames) > labels.size:
                logger.warn(f"{video_name}:# of frames ({len(frames)}) not equal to # of labels ({labels.size})")
                frames = frames[:labels.size]
        
        # get example segments
        class_vid_locs = example_video_segments(labels, min_bout_len, n_examples)
        all_videos.append({
            "segments": class_vid_locs, 
            "frames": frames
            })
    
    n_classes = len(all_videos[0]["segments"])
    for video_data in all_videos:
        if len(video_data["segments"]) != n_classes:
            curr_classes = len(video_data["segments"])
            logger.warn(f"number of groups not consistent across test animals ({curr_classes}/{n_classes})")

    frame = cv2.imread(all_videos[0]["frames"][0])
    height, width, layers = frame.shape

    logger.info(f"saving example videos from {n_classes} classes to {outdir}")
    for k in range(n_classes):
        logger.info(f"generating example videos for class {k}")
        video_name = os.path.join(outdir, 'group_{}.mp4'.format(k))
        video_frames = []

        for video_data in all_videos:
            curr_frames = video_data["frames"]
            curr_vid_locs = video_data["segments"][k]

            for j, vid in enumerate(curr_vid_locs):
                for idx in range(vid['start'], vid['end']+1):
                    video_frames.append(cv2.imread(curr_frames[idx]))
                for idx in range(bsoid.fps):
                    video_frames.append(np.zeros(shape=(height, width, layers), dtype=np.uint8))
            
        videomaker(video_frames, int(bsoid.fps), video_name)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    bsoid = BSOID("./config/config.yaml")
    raw_file = "../../data/videos/WT009G2NCIN20825M-25-PSY_pose_est_v2.h5"
    video_file = "../../data/videos/WT009G2NCIN20825M-25-PSY.avi"

    fdata, frames = extract_data_from_video(bsoid, raw_file, video_file)
    labels = labels_for_video(bsoid, fdata)

    print(labels.shape)