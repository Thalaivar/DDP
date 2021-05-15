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

def cluster_with_hdbscan(feats, cluster_range, HDBSCAN_PARAMS, verbose=False):
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

        if max_entropy(numulab[-1]) != 0:
            e_ratio = -sum([p*np.log2(p) for p in prop])/max_entropy(numulab[-1])
        else:
            e_ratio = 0
        entropy.append(e_ratio)

        # logging.info(f'identified {numulab[-1]} clusters (max is {max(numulab)}) with min_sample_prop={round(min_c, 2)} and entropy_ratio={round(entropy[-1], 3)}')
        
        # # retain max_clusters
        # if numulab[-1] > highest_numulab:
        #     highest_numulab = numulab[-1]
        #     best_clf = trained_classifier

        # # retain best distribution
        # if numulab[-1] == highest_numulab and entropy[-1] > highest_entropy:
        #     highest_entropy = entropy[-1]
        #     best_clf = trained_classifier

        if verbose:
            logger.info(f"identified {numulab[-1]} clusters with min_sample_prop={round(min_c,2)} and entropy ratio={round(entropy[-1], 3)}")
        if entropy[-1] > highest_entropy:
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

def calculate_entropy_ratio(labels):
    prop = [p / labels.size for p in np.unique(labels, return_counts=True)[1]]
    entropy_ratio = -sum(p * np.log2(p) for p in prop) / max_entropy(labels.max() + 1)
    return entropy_ratio

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