import re
import os
import cv2
import ffmpeg
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix

def create_confusion_matrix(feats, labels, clf):
    pred = clf.predict(feats)
    data = confusion_matrix(labels, pred, normalize='all')
    df_cm = pd.DataFrame(data, columns=np.unique(pred), index=np.unique(pred))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, cmap="Blues", annot=False)
    plt.show()

    return data

def frames_from_video(video_file, frame_dir):
    probe = ffmpeg.probe(video_file)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])
    bit_rate = int(video_info['bit_rate'])
    avg_frame_rate = round(int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(video_info['avg_frame_rate'].rpartition('/')[2]))
    logging.info('frame extraction for {} frames at {} frames per second'.format(num_frames, avg_frame_rate))
    try:
        (ffmpeg.input(video_file)
            .filter('fps', fps=avg_frame_rate)
            .output(str.join('', (frame_dir, '/frame%01d.png')), video_bitrate=bit_rate,
                    s=str.join('', (str(int(width * 0.5)), 'x', str(int(height * 0.5)))), sws_flags='bilinear',
                    start_number=0)
            .run(capture_stdout=True, capture_stderr=True))
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))

def get_all_bouts(labels):
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

def create_vids(labels, frame_dir, output_path, temporal_window, bout_length, n_examples, output_fps):
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    images.sort(key=lambda x:alphanum_key(x))
    
    # trim frames since we exclude first and last few frames when taking fft
    images = images[temporal_window // 2:-temporal_window // 2 + 1]
    logging.debug(f'using {len(images)} frames for creating example videos')

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape

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
    
    for i, class_vids in enumerate(class_vid_locs):
        video_name = 'group_{}.mp4'.format(i)
        video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, output_fps, (width, height))
        video_frames = []
        for j, vid in enumerate(class_vids):
            for idx in range(vid['start'], vid['end']+1):
                video_frames.append(images[idx])
            for idx in range(output_fps):
                video_frames.append(np.zeros(shape=(height, width, layers), dtype=np.uint8))
        for image in video_frames:
            if isinstance(image, str):
                video.write(cv2.imread(os.path.join(frame_dir, image)))
            elif isinstance(image, np.ndarray):
                video.write(image)
        cv2.destroyAllWindows()
        video.release()
    
    return

def alphanum_key(s):
    
    def convert_int(s):
        if s.isdigit():
            return int(s)
        else:
            return s

    return [convert_int(c) for c in re.split('([0-9]+)', s)]