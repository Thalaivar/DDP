import enum
import os
import cv2
import h5py
import logging
import joblib

import numpy as np
import pandas as pd
from tqdm import tqdm
from BSOID.bsoid import BSOID
from itertools import combinations
from BSOID.utils import alphanum_key
from BSOID.data import process_h5py_data, bsoid_format
from BSOID.features import extract_comb_feats, aggregate_features
from BSOID.preprocessing import smoothen_data, likelihood_filter, windowed_feats

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
        extract_frames_from_video(video, frame_dir)
    
    # get keypoint data from video
    conf, pos = process_h5py_data(h5py.File(raw_data_file, "r"))
    bsoid_data = bsoid_format(conf, pos)
    fdata, perc_filt = likelihood_filter(bsoid_data, bsoid.fps, bsoid.conf_threshold, bsoid.bodyparts, **bsoid.trim_params)
    if perc_filt > bsoid.filter_thresh:
        logger.warning(f"% data filtered from {os.path.split(raw_data_file)[-1]} too high ({perc_filt}%)")

    frames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir)], key=alphanum_key)
    frames = trim_frames(frames, bsoid.fps, **bsoid.trim_params)
    assert len(frames) == fdata['x'].shape[0], f"video:{video_name}: # of frames ({len(frames)}) does not match with no. of datapoints ({fdata['x'].shape[0]})"
    
    return fdata, frames

def trim_frames(frames, fps, end_trim, clip_window):
    # baseline video only 
    HOUR_LEN = 55 * 60 * fps
    frames = frames[:HOUR_LEN]
    
    if end_trim > 0:
        end_trim *= (fps * 60)
        frames = frames[end_trim:-end_trim]

    if clip_window > 0:
            # take first clip_window after trimming
            clip_window *= (60 * fps)
            frames = frames[end_trim:end_trim + clip_window]

    return frames

def extract_frames_from_video(video: cv2.VideoCapture, frame_dir: str):
    fps = video.get(cv2.CAP_PROP_FPS)
    success, image = video.read()
    frame_idx = 0
    while success:
        cv2.imwrite(os.path.join(frame_dir, f"frame{frame_idx}.jpg"), image)
        frame_idx += 1
        success, image = video.read()
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

def predict_frames(fdata, fps, stride_window, clf):
    def extract(fdata, fps, stride_window):
        x, y = fdata['x'], fdata['y']
        assert x.shape == y.shape
        N, n_dpoints = x.shape

        win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
        
        disp = np.linalg.norm(np.array([x[1:,:] - x[0:N-1,:], y[1:,:] - y[0:N-1,:]]), axis=0)
        links = [np.array([x[:,i] - x[:,j], y[:,i] - y[:,j]]).T for i, j in combinations(range(n_dpoints), 2)]
        link_angles = np.vstack([np.arctan2(link[:,1], link[:,0]) for link in links]).T
        ll = np.vstack([np.linalg.norm(link, axis=1) for link in links]).T
        dis_angles = np.vstack([np.arctan2(np.cross(link[0:N-1], link[1:]), np.sum(link[0:N-1] * link[1:], axis=1)) for link in links]).T
        
        for i in range(ll.shape[1]):
            ll[:,i] = smoothen_data(ll[:,i], win_len)
            dis_angles[:,i] = smoothen_data(dis_angles[:,i], win_len)
            link_angles[:,i] = smoothen_data(link_angles[:,i], win_len)
        for i in range(disp.shape[1]):
            disp[:,i] = smoothen_data(disp[:,i], win_len)

        fs_feats = []
        for i in range(stride_window):
            win_ll = windowed_feats(ll[i:], stride_window, mode="mean")
            win_link_angles = windowed_feats(link_angles[i:], stride_window, mode="mean")
            win_disp = windowed_feats(disp[i:], stride_window, mode="sum")
            win_dis_angles = windowed_feats(dis_angles[i:], stride_window, mode="sum")

            if win_ll.shape[0] != win_disp.shape[0]:
                if win_ll.shape[0] - 1 == win_disp.shape[0]:
                    win_ll, win_link_angles = win_ll[1:], win_link_angles[1:]
                else:
                    raise ValueError(f"incorrect shapes for geometric {win_ll.shape} and displacement {win_disp.shape} features")
                    
            fs_feats.append(np.hstack((win_ll, win_link_angles, win_dis_angles, win_disp)))
        
        return fs_feats
    
    fs_feats = extract(fdata, fps, stride_window)
    fs_labels = [clf.predict(f).squeeze() for f in fs_feats]

    max_len = max([f.shape[0] for f in fs_labels])
    for i, f in enumerate(fs_labels):
        pad_arr = -1 * np.ones((max_len,))
        pad_arr[:f.shape[0]] = f
        fs_labels[i] = pad_arr
    labels = np.array(fs_labels).flatten('F')
    labels = labels[labels >= 0]

    return labels

def labels_for_video(bsoid, rawfile, vid_file, extract_frames=False):
    filtered_data, frames = extract_data_from_video(bsoid, rawfile, vid_file, extract_frames)
    geom_feats = extract_comb_feats(filtered_data, bsoid.fps)
    
    stride_window = 3

    fs_feats = []
    for i in range(stride_window):
        fs_feats.append(aggregate_features(geom_feats[i:], stride_window))

    with open("D:/IIT/DDP/data/tests/test.model", "rb") as f:
        clf = joblib.load(f)

    fs_labels = [clf.predict(f).squeeze() for f in fs_feats]
    max_len = max([f.shape[0] for f in fs_labels])
    for i, f in enumerate(fs_labels):
        pad_arr = -1 * np.ones((max_len,))
        pad_arr[:f.shape[0]] = f
        fs_labels[i] = pad_arr
    labels = np.array(fs_labels).flatten('F')
    labels = labels[labels >= 0]

    return labels, frames

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

def add_group_label_to_frame(frames, label):
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.6
    # parameters for adding text
    text = f'Group {label}'
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    text_offset_x = 20
    text_offset_y = 20
    box_coords = ((text_offset_x - 12, text_offset_y + 12), (text_offset_x + text_width + 12, text_offset_y - text_height - 8))

    for i, frame in enumerate(frames):
        frames[i] = cv2.rectangle(frame, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
        frames[i] = cv2.putText(frame, text, (text_offset_x, text_offset_y), font,
                            fontScale=font_scale, color=(255, 255, 255), thickness=1)
    
    return frames

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
                logger.warning(f"{video_name}:# of frames ({len(frames)}) not equal to # of labels ({labels.size})")
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
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    video_name = os.path.join(outdir, "examples.mp4")
    video = cv2.VideoWriter(video_name, fourcc, int(bsoid.fps), (width, height))

    logger.info(f"saving example videos from {n_classes} classes to {video_name}")
    for k in tqdm(range(n_classes)):
        class_video_frames = []

        for video_data in all_videos:
            curr_frames = video_data["frames"]
            curr_vid_locs = video_data["segments"][k]

            for j, vid in enumerate(curr_vid_locs):
                video_frames = []
                for idx in range(vid['start'], vid['end']+1):
                    video_frames.append(cv2.imread(curr_frames[idx]))
                video_frames = add_group_label_to_frame(video_frames, k)
                class_video_frames.extend(video_frames)
                for idx in range(bsoid.fps):
                    class_video_frames.append(np.zeros(shape=(height, width, layers), dtype=np.uint8))
        
        if len(class_video_frames) > 0:
            for frame in class_video_frames:
                video.write(frame.astype(np.uint8))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    bsoid = BSOID("./config/config.yaml")
    video_dir = "../../data/tests/"

    create_class_examples(bsoid, video_dir, min_bout_len=200, n_examples=5, outdir=video_dir)