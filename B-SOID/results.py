import matplotlib as mpl
mpl.use('tKAgg')

import os
import logging
from BSOID.bsoid import BSOID

logging.basicConfig(level=logging.INFO)

base_dir = 'D:/IIT/DDP/data'
bsoid = BSOID.load_config(base_dir, 'split')

video_dir = bsoid.test_dir + '/videos'
csv_dir = bsoid.test_dir
bsoid.create_examples(csv_dir, video_dir, bout_length=3, n_examples=10)