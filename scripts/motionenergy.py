import random

from tqdm import tqdm
from behaviourPipeline.prediction import *
from behaviourPipeline.pipeline import BehaviourPipeline

def poses_for_bouts(fdata, locs):
    data = np.array([fdata['x'], fdata['y']])
    data = [data[:,loc.start:loc.end+1,:] for loc in locs]
    return data
    
def bout_pose_data(bout_idx, videos, min_bout_len, fps, n_examples, clf, stride_window, bodyparts, conf_threshold, filter_thresh):
    min_bout_len = fps * min_bout_len // 1000
    bout_data = []
    
    for video_file, _ in videos:
        fdata = extract_data_from_video(video_file, bodyparts, fps, conf_threshold, filter_thresh)
        labels = video_frame_predictions(video_file, clf, stride_window, bodyparts, fps, conf_threshold, filter_thresh)
        locs = bouts_from_video(bout_idx, labels, min_bout_len, n_examples)    
        bout_data.extend(poses_for_bouts(fdata, locs))
    
    return bout_data

def main(pipeline, video_dirs, min_bout_len, n_examples):
    clf = pipeline.load("classifier.sav")
    max_label = clf.classes_.max()
    
    n = len(video_dirs) // (max_label + 1)
    if n < 1: raise ValueError("need more videos to have at least one unique video per behaviour")
    random.shuffle(video_dirs)
    
    j, bout_data = 0, []
    for i in tqdm(range(n, len(video_dirs), n)):
        bout_data.append(
            bout_pose_data(
                j,
                video_dirs[i-n:i], 
                min_bout_len, 
                pipeline.fps, 
                n_examples, 
                clf, 
                pipeline.stride_window,
                pipeline.bodyparts,
                pipeline.conf_threshold,
                pipeline.filter_thresh
            )
        )
    
    return bout_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("motionenergy.py")
    parser.add_argument("--video-list", type=str, required=True)
    parser.add_argument("--pipelinename", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    
    args = parser.parse_args()
    with open(args.video_list, 'r') as f:
        videos = [s.rstrip().split(',') for s in f.readlines()]
    
    pipeline = BehaviourPipeline(args.pipelinename, args.config)
    bout_data = main(pipeline, videos, min_bout_len=200, n_examples=5)