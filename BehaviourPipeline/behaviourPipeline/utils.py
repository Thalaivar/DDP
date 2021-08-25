import re
import warnings
import logging
import numpy as np
import pandas as pd
import ftplib
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

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
    n = labels.max() + 1
    prop = [p / labels.size for p in np.unique(labels, return_counts=True)[1]]

    if max_entropy(n) != 0:
        entropy_ratio= -sum([p*np.log2(p) for p in prop])/max_entropy(n)
    else:
        entropy_ratio = 0

    return entropy_ratio

def get_random_video_and_keypoints(data_file, save_dir):
    data = pd.read_csv(data_file)
    
    session = ftplib.FTP("ftp.box.com")
    from getpass import getpass
    password = getpass("Box login password: ")
    session.login("ae16b011@smail.iitm.ac.in", password)

    data = dict(data.iloc[np.random.randint(0, data.shape[0], 1)[0]])
    data_filename, vid_filename = get_video_and_keypoint_data(session, data, save_dir)
    session.quit()

    return data_filename, vid_filename

def bootstrap_estimate(data, n, ns):
    N = data.shape[0]
    if N < n:
        warnings.warn(f"Number of samples to be drawn is greater than population")
        return np.median(data, axis=0), data.std(axis=0)
    
    data = np.array([np.mean(data[np.random.choice(N, n, replace=True)], axis=0) for _ in range(ns)])
    return data.mean(axis=0), data.std(axis=0)

def normalize_statistics(data):
    for i in range(data.shape[0]):
        if data[i].sum() > 0:
            data[i] /= data[i].sum()
    
    return data[i]
    
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