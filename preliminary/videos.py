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
        for j, vid in enumerate(class_vids):
            video_name = 'group_{}_example_{}.mp4'.format(i, j)
            video_frames = []
            for idx in range(vid['start'], vid['end']+1):
                video_frames.append(images[idx])
            video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, output_fps, (width, height))
            for image in video_frames:
                video.write(cv2.imread(os.path.join(frame_dir, image)))
            cv2.destroyAllWindows()
            video.release()
    
    return