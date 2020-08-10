import h5py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import ftplib
from LOCAL_CONFIG import *

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

# retain the middle 30 min of the video
RETAIN_WINDOW = 30*60

def conv_bsoid_format(filename):
    f = h5py.File(filename, "r")

    data = list(f.keys())[0]
    keys = list(f[data].keys())

    conf = np.array(f[data][keys[0]])
    pos = np.array(f[data][keys[1]])

    # read in the timestamps
    # txt_fname = filename[0:-14] + 'timestamps.txt'
    # with open(txt_fname) as f:
    #     tstmps = np.array(f.readlines(), dtype=float)

    # no. of frames to retain
    n_frames = FPS*RETAIN_WINDOW
    n_frames += n_frames%2
    # index to middle of data
    mid_idx = round(conf.shape[0]/2)
    # retain only middle 30 mins of data
    conf = conf[int(mid_idx-n_frames/2):int(mid_idx+n_frames/2)+1, :]
    pos = pos[int(mid_idx-n_frames/2):int(mid_idx+n_frames/2)+1, :]

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
    
    bsoid_data_headers = []
    bodypart_headers = []
    for i in range(len(BSOID_DATA_INDICES)):
        bsoid_data_headers.append('likelihood')
        bsoid_data_headers.append('x')
        bsoid_data_headers.append('y')

        bodypart_headers.append(BSOID_DATA[i])
        bodypart_headers.append(BSOID_DATA[i])
        bodypart_headers.append(BSOID_DATA[i])
    
    # hacky fix to get it in B-SOID format
    bsoid_data = np.vstack((np.array(bsoid_data_headers), bsoid_data))
    bsoid_data = np.vstack((np.array(bodypart_headers), bsoid_data))
    idx = np.arange(bsoid_data.shape[0]).reshape(bsoid_data.shape[0], 1)
    bsoid_data = np.hstack((idx, bsoid_data))
    
    bsoid_data = pd.DataFrame(bsoid_data)
    bsoid_data.to_csv(filename[:-3] +'.csv', index=False)

    return bsoid_data

def download_data(bsoid_data_file, pose_est_dir):
    bsoid_data = pd.read_csv(bsoid_data_file)
    bsoid_data = bsoid_data.sample(n=30)
    bsoid_data = bsoid_data.reset_index()
    
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
        session.retrbinary("RETR "+ filename, open(pose_est_dir + filename, 'wb').write)
        # filename = movie_name[0:-4] + "_timestamps.txt"
        # session.retrbinary("RETR "+ filename, open(pose_est_dir + filename, 'wb').write)
        session.cwd('/')

if __name__ == "__main__":
    # to download files
    # download_data("bsoid_strain_data.csv", '/home/ubuntu/data/train/')

    # to convert files to csv (call from folder containing .h5 files)
    file_path = "/home/ubuntu/data/train/"
    files = os.listdir(file_path)

    for i in tqdm(range(len(files))):
        conv_bsoid_format(file_path+files[i])