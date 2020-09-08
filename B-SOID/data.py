import h5py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import ftplib
import random

# Mapping for data
NOSE_INDEX = 0
LEFT_EAR_INDEX = 1
RIGHT_EAR_INDEX = 2
BASE_NECK_INDEX = 3
LEFT_FRONT_PAW_INDEX = 4
RIGHT_FRONT_PAW_INDEX = 5
CENTER_SPINE_INDEX = 6
LEFT_REAR_PAW_INDEX = 7
RIGHT_REAR_PAW_INDEX = 8
BASE_TAIL_INDEX = 9
MID_TAIL_INDEX = 10
TIP_TAIL_INDEX = 11

RETAIN_WINDOW = 30*60
FPS = 30


def conv_bsoid_format(filename, save_dir, clip_data=True):
    f = h5py.File(filename, "r")
    # retain only filename
    filename = filename.split('/')[-1]

    data = list(f.keys())[0]
    keys = list(f[data].keys())

    conf = np.array(f[data][keys[0]])
    pos = np.array(f[data][keys[1]])

    if clip_data:
        n_frames = FPS*RETAIN_WINDOW
        n_frames += n_frames%2
        # index to middle of data
        mid_idx = conf.shape[0] // 2
        # retain only middle 30 mins of data
        conf = conf[int(mid_idx-n_frames/2)-1:int(mid_idx+n_frames/2)+1, :]
        pos = pos[int(mid_idx-n_frames/2)-1:int(mid_idx+n_frames/2)+1, :]

    # retreive the data for B-SOiD
    BSOID_DATA = ['HEAD', 'BASE_NECK', 'CENTER_SPINE', 'HINDPAW1', 'HINDPAW2', 'MID_TAIL', 'TIP_TAIL']
    BSOID_DATA_INDICES = [0, BASE_NECK_INDEX, CENTER_SPINE_INDEX, RIGHT_REAR_PAW_INDEX, 
                          LEFT_REAR_PAW_INDEX, MID_TAIL_INDEX, TIP_TAIL_INDEX]

    bsoid_data = np.zeros((conf.shape[0], 3*len(BSOID_DATA_INDICES)))
    i = 0
    for bodypart in BSOID_DATA_INDICES:
        if bodypart == 0:
            bsoid_data[:,i] = (conf[:,NOSE_INDEX] + conf[:,LEFT_EAR_INDEX] + conf[:,RIGHT_EAR_INDEX])/3
            bsoid_data[:,i+1] = (pos[:,NOSE_INDEX,0] + pos[:,LEFT_EAR_INDEX,0] + pos[:,RIGHT_EAR_INDEX,0])/3
            bsoid_data[:,i+2] = (pos[:,NOSE_INDEX,1] + pos[:,LEFT_EAR_INDEX,1] + pos[:,RIGHT_EAR_INDEX,1])/3
        else:
            bsoid_data[:,i] = conf[:,bodypart]
            bsoid_data[:,i+1] = pos[:,bodypart,0]
            bsoid_data[:,i+2] = pos[:,bodypart,1]
        i += 3
    
    bodypart_headers = []
    for i in range(len(BSOID_DATA_INDICES)):
        bodypart_headers.append(BSOID_DATA[i]+'_lh')
        bodypart_headers.append(BSOID_DATA[i]+'_x')
        bodypart_headers.append(BSOID_DATA[i]+'_y')
    
    bsoid_data = pd.DataFrame(bsoid_data)
    bsoid_data.columns = bodypart_headers

    bsoid_data.to_csv(save_dir + '/' + filename[:-3] +'.csv', index=False)

    return bsoid_data

def download_data(bsoid_data_file, pose_est_dir):
    bsoid_data = pd.read_csv(bsoid_data_file)
    
    session = ftplib.FTP("ftp.box.com")
    session.login("ae16b011@smail.iitm.ac.in", "Q0w9e8r7t6Y%Z")

    strains = ["LL6-B2B", "LL5-B2B", "LL4-B2B", "LL3-B2B", "LL2-B2B", "LL1-B2B"]
    datasets = ["strain-survey-batch-2019-05-29-e/", "strain-survey-batch-2019-05-29-d/", "strain-survey-batch-2019-05-29-c/",
                "strain-survey-batch-2019-05-29-b/", "strain-survey-batch-2019-05-29-a/"]

    # master directory where datasets are saved
    master_dir = 'JAX-IITM Shared Folder/Datasets/'
    n_data = bsoid_data.shape[0]
    print("Downloading data sets from box...")
    for i in tqdm(range(n_data)):
        strain, data, movie_name = bsoid_data['NetworkFilename'][i].split('/')
        
        # change to correct box directory
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
        filename = movie_name[0:-4] + "_pose_est_v2.h5"
        session.retrbinary("RETR "+ filename, open(pose_est_dir + '/' + filename, 'wb').write)
        session.cwd('/')