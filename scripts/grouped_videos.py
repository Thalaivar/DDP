import os
import cv2
import joblib
import logging
from prediction import *
from BSOID.bsoid import BSOID
from BSOID.analysis import get_group_map

logger = logging.getLogger(__name__)

def grouped_example_clips(config_file, outdir, video_dir, model_file, eac_mat_file, pvalue=0.53, min_bout_len=200, n_examples=5):
    bsoid = BSOID(config_file)

    with open(model_file, "rb") as f:
        clf = joblib.load(f)
    
    min_bout_len = bsoid.fps * min_bout_len // 1000

    try: os.mkdir(outdir)
    except FileExistsError: pass

    video_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".avi")])
    raw_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".h5")])

    all_videos = []
    for raw_file, video_file in zip(raw_files, video_files):
        video_name = os.path.split(video_file)[-1][:-4]
        labels, frames = labels_for_video(bsoid, clf, raw_file, video_file)
        
        if labels.size != len(frames):
            if len(frames) > labels.size:
                logger.warning(f"{video_name}:# of frames ({len(frames)}) not equal to # of labels ({labels.size})")
                frames = frames[:labels.size]
        
        # get example segments
        class_vid_locs = example_video_segments(labels, min_bout_len, n_examples)
        all_videos.append({
            "segments": class_vid_locs, 
            "frames": frames
            })
    
    n_classes = len(all_videos[0]["segments"])
    for video_data in all_videos:
        if len(video_data["segments"]) != n_classes:
            curr_classes = len(video_data["segments"])
            logger.warn(f"number of groups not consistent across test animals ({curr_classes}/{n_classes})")

    frame = cv2.imread(all_videos[0]["frames"][0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    group_map, _, _ = get_group_map(eac_mat_file, pvalue)
    group_clips = {glab: [] for glab in group_map.keys()}
    for glab, motifs in group_map.items():
        for k in motifs:
            class_video_frames = []
            for video_data in all_videos:
                curr_frames = video_data["frames"]
                curr_vid_locs = video_data["segments"][k]

                for j, vid in enumerate(curr_vid_locs):
                    video_frames = []
                    for idx in range(vid['start'], vid['end']+1):
                        video_frames.append(cv2.imread(curr_frames[idx]))
                    video_frames = add_group_label_to_frame(video_frames, k)
                    class_video_frames.extend(video_frames)
                    for idx in range(bsoid.fps):
                        class_video_frames.append(np.zeros(shape=(height, width, layers), dtype=np.uint8))                
            group_clips[glab].extend(class_video_frames)

    for glab, clip_frames in group_clips.items():
        video_name = os.path.join(outdir, f"group-{glab}.mp4")
        video = cv2.VideoWriter(video_name, fourcc, int(bsoid.fps), (width, height))

        if len(clip_frames) > 0:
            for frame in clip_frames:
                video.write(frame.astype(np.uint8))

        cv2.destroyAllWindows()
        video.release()

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("grouped_videos.py")
    parser.add_argument("--config", type=str, help="configuration file for B-SOID")
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--video-dir", type=str)
    parser.add_argument("--model-file", type=str)
    parser.add_argument("--eac-file", type=str)
    parser.add_argument("--pvalue", type=float, default=0.53)
    parser.add_argument("--min-bout-len", type=int, default=200)
    parser.add_argument("--n-examples", type=int, default=5)
    args = parser.parse_args()

    import logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO)

    grouped_example_clips(
            args.config, 
            args.outdir,
            args.video_dir,
            args.model_file,
            args.eac_file,
            args.pvalue,
            args.min_bout_len,
            args.n_examples
        )