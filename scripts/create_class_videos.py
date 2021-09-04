import os
from behaviourPipeline.pipeline import BehaviourPipeline

import argparse
parser = argparse.ArgumentParser("create_class_videos.py")
parser.add_argument("--video-list", type=str, required=True)
parser.add_argument("--pipelinename", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--outdir", type=str, required=True)

args = parser.parse_args()
with open(args.video_list, 'r') as f:
    videos = [s.rstrip().split(',') for s in f.readlines()]


pipeline = BehaviourPipeline(args.pipelinename, args.config)
pipeline.create_example_videos(videos, min_bout_len=200, n_examples=5, outdir=args.outdir) 