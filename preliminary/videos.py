import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from utils import sort_nicely

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

def create_vids(labels, crit, counts, output_fps, frame_dir, output_path):
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape

    class_vid_locs = get_all_bouts(labels)

    # filter out all videos smaller than len crit
    for k, class_vids in enumerate(class_vid_locs):
        for i, vid in enumerate(class_vids):
            if vid['end'] - vid['start'] < crit:
                del class_vid_locs[k][i]
    
    # get longest vids
    for k, class_vids in enumerate(class_vid_locs):
        class_vids.sort(key=lambda loc: loc['start']-loc['end'])
        if len(class_vids) > counts:
            class_vid_locs[k] = class_vids[0:counts]
    
    for i, class_vids in enumerate(tqdm(class_vid_locs)):
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