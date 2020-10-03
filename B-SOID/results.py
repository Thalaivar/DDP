import matplotlib as mpl
mpl.use('tKAgg')

import os
import logging
from BSOID.bsoid import BSOID

logging.basicConfig(level=logging.INFO)

base_dir = 'D:/IIT/DDP/data'
bsoid = BSOID.load_config(base_dir, 'geo_feats')

video_dir = bsoid.test_dir + '/videos'
vid_files = [video_dir + '/' + f for f in os.listdir(video_dir) if f.endswith('.avi')]
csv_files = [bsoid.test_dir + '/' + f for f in os.listdir(bsoid.test_dir) if f.endswith('.csv')]
vid_files.sort()
csv_files.sort()

for i in range(len(vid_files)):
    bsoid.label_frames(csv_files[i], vid_files[i], extract_frames=False, load_feats=False, bout_length=3, n_examples=10, output_fps=30)