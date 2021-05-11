import os
import cv2
import joblib
import pandas as pd

from BSOID.utils import *
from BSOID.bsoid import BSOID
from prediction import *
from BSOID.preprocessing import likelihood_filter

import logging
logging.basicConfig(level=logging.INFO)

def create_labelled_vids():
    video_length = 30
    output_fps = 30
    
    bsoid = BSOID.load_config(base_dir='D:/IIT/DDP/data', run_id='dis_active')
    vid_dir = bsoid.test_dir + '/videos'
    csv_dir = bsoid.test_dir
    output_path = bsoid.base_dir + '/' + bsoid.run_id + '_labelled_vids'
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass

    csv_files = [csv_dir + '/' + f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    video_files = [vid_dir + '/' + f for f in os.listdir(vid_dir) if f.endswith('.avi')]
    csv_files.sort()
    video_files.sort()

    video_length *= 60*bsoid.fps
    n_animals = len(csv_files)
    logging.info(f'labelling videos of length {video_length // (60*bsoid.fps)} mins from {n_animals} animals')

    for i in range(n_animals):
        video_name = csv_files[i].split('/')[-1][-4]
        labels, frame_dir = predict_labels(bsoid, csv_files[i], video_files[i], video_length)
        labelled_vids(labels, frame_dir, output_path, video_length, video_name, output_fps)

    
    try:
        os.mkdir(output_path)
    except FileExistsError:
        for f in os.listdir(output_path):
            os.remove(output_path + '/' + f)

def labelled_vids(labels, frame_dir, output_path, video_length, video_name, output_fps):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]    
    images.sort(key=lambda x:alphanum_key(x))

    mid_idx = len(images) // 2
    images = images[mid_idx - video_length // 2:mid_idx + video_length // 2 + 1]

    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, _ = frame.shape
    video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, output_fps, (width, height))
    for j in range(len(labels)):
        frame = add_group_label_to_frame(os.path.join(frame_dir, images[j]), labels[j])
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def add_group_label_to_frame(path_to_frame, label):
    label = f'Group {label}'
    frame = cv2.imread(path_to_frame)
    height, width, layers = frame.shape

    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    # parameters for adding text
    text = f'Group {label}'
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    text_offset_x = 50
    text_offset_y = 50
    box_coords = ((text_offset_x - 12, text_offset_y + 12), (text_offset_x + text_width + 12, text_offset_y - text_height - 8))
    cv2.rectangle(frame, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
    cv2.putText(frame, text, (text_offset_x, text_offset_y), font,
                            fontScale=font_scale, color=(255, 255, 255), thickness=1)
    
    return frame

def predict_labels(bsoid: BSOID, csv_file, video_file, video_length):
    # directory to store results for video
    output_dir = bsoid.test_dir + '/' + csv_file.split('/')[-1][:-4]
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    
    frame_dir = output_dir + '/pngs'
    extract_frames = True
    try:
        os.mkdir(frame_dir)
    except FileExistsError:
        extract_frames = False

    if extract_frames:
        logging.info('extracting frames from video {} to dir {}'.format(video_file, frame_dir))
        frames_from_video(video_file, frame_dir)
    
    logging.debug('extracting features from {}'.format(csv_file))
    data = pd.read_csv(csv_file, low_memory=False)
    
    mid_idx = data.shape[0] // 2
    data, _ = likelihood_filter(data)
    data = data[mid_idx - video_length // 2:mid_idx + video_length // 2 + 1]

    feats = frameshift_features(data, bsoid.stride_window, bsoid.fps, extract_feats, window_extracted_feats, bsoid.temporal_window, bsoid.temporal_dims)
    with open(bsoid.output_dir + '/' + bsoid.run_id + '_classifiers.sav', 'rb') as f:
        clf = joblib.load(f)

    labels = frameshift_predict(feats, clf, bsoid.stride_window)
    logging.info(f'predicted {len(labels)} frames in {feats[0].shape[1]}D with trained classifier')

    return labels, frame_dir

def extract_frames_from_all_videos():
    from joblib import Parallel, delayed

    bsoid = BSOID("D:/IIT/DDP/DDP/B-SOID/config/config.yaml")
    video_dir = "D:/IIT/DDP/data/videos"

    video_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".avi")])
    raw_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".h5")])
    
    Parallel(n_jobs=1)(delayed(extract_data_from_video)(bsoid, raw_file, video_file) for raw_file, video_file in zip(raw_files, video_files))

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # extract_frames_from_all_videos()

    bsoid = BSOID("D:/IIT/DDP/DDP/B-SOID/config/config.yaml")
    video_dir = "D:/IIT/DDP/data/videos"
    results_dir=f"{video_dir}/{bsoid.run_id}_results"
    create_class_examples(bsoid, video_dir, min_bout_len=200, n_examples=10, outdir=results_dir)
    