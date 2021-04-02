import os
import cv2
import joblib
import numpy as np
import pandas as pd

SAVE_DIR = "D:/IIT/DDP/data/paper/figure2"
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

