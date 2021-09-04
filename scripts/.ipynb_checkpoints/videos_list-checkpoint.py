import os
import argparse
from behaviourPipeline.preprocessing import get_filename_in_dataset

parser = argparse.ArgumentParser("create_class_videos.py")
parser.add_argument("--videos-dir", type=str, required=True)
parser.add_argument("--data-dir", type=str, required=True)
args = parser.parse_args()

videos_dir = args.videos_dir
dirs = [f for f in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir, f))]
dirs2 = []
for dir in dirs:
    dirs2.extend(
        [
            os.path.join(dir, f) 
            for f in os.listdir(os.path.join(videos_dir, dir)) 
            if os.path.isdir(os.path.join(videos_dir, dir, f))
        ]
    )

video_files = []
for dir in dirs2:
    video_files.extend(
        [
            os.path.join(dir, f)
            for f in os.listdir(os.path.join(videos_dir, dir))
            if f.endswith(".avi")
        ]
    )
    
data_files = [get_filename_in_dataset(args.data_dir, f) for f in video_files]
video_files = [os.path.join(videos_dir, f) for f in video_files]

with open("videoList.txt", 'w') as videolistfile:
    for data_file, video_file in zip(data_files, video_files):
        if os.path.exists(data_file) and os.path.exists(video_file):
            videolistfile.write(f"{data_file},{video_file}\n")