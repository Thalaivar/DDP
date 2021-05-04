import re
import os
try:
    import cv2
    import ffmpeg
except:
    pass
try:
    import hdbscan
except:
    pass
import logging
import numpy as np
import pandas as pd
import ftplib
try:
    import matplotlib.pyplot as plt
except:
    pass
try:
    import seaborn as sn
except:
    pass
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

def cluster_with_hdbscan(feats, cluster_range, HDBSCAN_PARAMS):
    highest_numulab, highest_entropy = -np.infty, -np.infty
    numulab, entropy = [], []
    if not isinstance(cluster_range, list):
        min_cluster_range = [cluster_range]
    elif len(cluster_range) == 2:
        min_cluster_range = np.linspace(*cluster_range, 25)
    elif len(cluster_range) == 3:
        cluster_range[-1] = int(cluster_range[-1])
        min_cluster_range = np.linspace(*cluster_range)
        
    for min_c in min_cluster_range:
        trained_classifier = hdbscan.HDBSCAN(min_cluster_size=int(round(min_c * 0.01 * feats.shape[0])),
                                            **HDBSCAN_PARAMS).fit(feats)
        
        labels = trained_classifier.labels_
        numulab.append(labels.max() + 1)
        prop = [0 for i in range(labels.max() + 1)]
        for i in range(labels.size):
            if labels[i] >= 0:
                prop[labels[i]] += 1
        prop = np.array(prop)
        prop = prop/prop.sum()
        entropy.append(-sum([p*np.log2(p) for p in prop])/max_entropy(numulab[-1]))

        logging.debug(f'identified {numulab[-1]} clusters (max is {max(numulab)}) with min_sample_prop={round(min_c, 2)} and entropy_ratio={round(entropy[-1], 3)}')
        
        # retain max_clusters
        if numulab[-1] > highest_numulab:
            highest_numulab = numulab[-1]
            best_clf = trained_classifier

        # retain best distribution
        if numulab[-1] == highest_numulab and entropy[-1] > highest_entropy:
            highest_entropy = entropy[-1]
            best_clf = trained_classifier

    assignments = best_clf.labels_
    soft_clusters = hdbscan.all_points_membership_vectors(best_clf)
    soft_assignments = np.argmax(soft_clusters, axis=1)

    return assignments, soft_clusters, soft_assignments, best_clf

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

def example_video_segments(labels, frame_dir, bout_length, n_examples, clip_len):
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    images.sort(key=lambda x:alphanum_key(x))

    # trim frames since we exclude first and last few frames when taking fft
    if clip_len is not None:
        images = images[clip_len:-clip_len]

    if len(labels) != len(images):
        logger.warn(f'# of labels {len(labels)} and frames {len(images)} do not match ; clipping images')
        images = images[:len(labels)]

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

    return class_vid_locs, images

def collect_all_examples(labels, frame_dirs, output_path, bout_length, n_examples, output_fps, clip_window):
    n_animals = len(frame_dirs)

    all_class_vid_locs = []
    all_frames = []
    for i in range(n_animals):
        class_vid_locs, frames = example_video_segments(labels[i], frame_dirs[i], bout_length, n_examples, clip_window)
        all_class_vid_locs.append(class_vid_locs)
        all_frames.append(frames)

    n_groups = len(all_class_vid_locs[0])
    for i in range(1, n_animals):
        if len(all_class_vid_locs[i]) != n_groups:
            logger.warn('number of groups not consistent across test animals')

    frame = cv2.imread(os.path.join(frame_dirs[0], all_frames[0][0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    for k in range(n_groups):
        video_name = 'group_{}.mp4'.format(k)
        video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, output_fps, (width, height))

        for i in range(n_animals):
            curr_frame_dir = frame_dirs[i]
            curr_frames = all_frames[i]
            curr_vid_locs = all_class_vid_locs[i][k]

            curr_video_frames = []
            for j, vid in enumerate(curr_vid_locs):
                for idx in range(vid['start'], vid['end']+1):
                    curr_video_frames.append(curr_frames[idx])
                for idx in range(output_fps):
                    curr_video_frames.append(np.zeros(shape=(height, width, layers), dtype=np.uint8))

            for image in curr_video_frames:
                if isinstance(image, str):
                    video.write(cv2.imread(os.path.join(curr_frame_dir, image)))
                elif isinstance(image, np.ndarray):
                    video.write(image)
        
        cv2.destroyAllWindows()
        video.release()

def create_vids(labels, frame_dir, output_path, clip_len, bout_length, n_examples, output_fps):
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    images.sort(key=lambda x:alphanum_key(x))
    
    # trim frames since we exclude first and last few frames when taking fft
    if clip_len is not None:
        images = images[clip_len:-clip_len + 1]
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

def max_entropy(n):
    probs = [1/n for _ in range(n)]
    return -sum([p*np.log2(p) for p in probs])

def get_random_video_and_keypoints(data_file, save_dir):
    data = pd.read_csv(data_file)
    
    session = ftplib.FTP("ftp.box.com")
    session.login("ae16b011@smail.iitm.ac.in", "rSNxWCBv1407")

    data = dict(data.iloc[np.random.randint(0, data.shape[0], 1)[0]])
    data_filename, vid_filename = get_video_and_keypoint_data(session, data, save_dir)
    session.quit()

    return data_filename, vid_filename


def get_video_and_keypoint_data(session, data, save_dir):
    strains = ["LL6-B2B", "LL5-B2B", "LL4-B2B", "LL3-B2B", "LL2-B2B", "LL1-B2B"]
    datasets = ["strain-survey-batch-2019-05-29-e/", "strain-survey-batch-2019-05-29-d/", "strain-survey-batch-2019-05-29-c/",
                "strain-survey-batch-2019-05-29-b/", "strain-survey-batch-2019-05-29-a/"]

    # master directory where datasets are saved
    master_dir = 'JAX-IITM Shared Folder/Datasets/'
    strain, data, movie_name = data['NetworkFilename'].split('/')

    idx = strains.index(strain)
    if idx == 0:
        movie_dir = master_dir + datasets[0] + strain + "/" + data + "/"
        session.cwd(movie_dir)
    elif idx == 5:
        movie_dir = master_dir + datasets[4] + strain + "/" + data + "/"
        session.cwd(movie_dir)
    else:
        try:
            movie_dir = master_dir + datasets[idx-1] + strain + "/" + data + "/"
            session.cwd(movie_dir)
        except:
            movie_dir = master_dir + datasets[idx] + strain + "/" + data + "/"
            session.cwd(movie_dir)

    # download data file
    data_filename = movie_name[0:-4] + "_pose_est_v2.h5"
    print(f"Downloading: {data_filename}")
    session.retrbinary("RETR "+ data_filename, open(save_dir + '/' + data_filename, 'wb').write)
    vid_filename = movie_name[0:-4] + ".avi"
    print(f"Downloading: {vid_filename}")
    session.retrbinary("RETR "+ vid_filename, open(save_dir + '/' + vid_filename, 'wb').write)

    return data_filename, vid_filename