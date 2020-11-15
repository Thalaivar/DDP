try:
    import h5py
except:
    pass
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import ftplib
import random

import logging
logger = logging.getLogger(__name__)

BSOID_DATA = ['NOSE', 'LEFT_EAR', 'RIGHT_EAR', 
        'BASE_NECK', 'FOREPAW1', 'FOREPAW2', 
        'CENTER_SPINE', 'HINDPAW1', 'HINDPAW2', 
        'BASE_TAIL', 'MID_TAIL', 'TIP_TAIL']

RETAIN_WINDOW = 30*60
FPS = 30

def extract_from_csv(filename, save_dir, trim=True):
    f = h5py.File(filename, "r")
    # retain only filename
    filename = filename.split('/')[-1]

    data = list(f.keys())[0]
    keys = list(f[data].keys())

    conf = np.array(f[data][keys[0]])
    pos = np.array(f[data][keys[1]])

    if trim:
        # trim first and last 2 minutes because there is no mouse
        END_TRIM = FPS*2*60
        conf, pos = conf[END_TRIM:-END_TRIM,:], pos[END_TRIM:-END_TRIM,:]

        clip_window = (FPS * RETAIN_WINDOW) // 2
        mid_idx = conf.shape[0] // 2
        clip_conf = conf[mid_idx-clip_window:mid_idx+clip_window,:]
        clip_pos = pos[mid_idx-clip_window:mid_idx+clip_window,:]
        bsoid_data = _bsoid_format(clip_conf, clip_pos)
        bsoid_data.to_csv(save_dir + f'/{filename[:-3]}.csv', index=False)

    else:
        bsoid_data = _bsoid_format(conf, pos)
        bsoid_data.to_csv(save_dir + '/' + filename[:-3] +'.csv', index=False)
        
def _bsoid_format(conf, pos):
    bsoid_data = np.zeros((conf.shape[0], 3*conf.shape[1]))

    j = 0
    for i in range(0, conf.shape[1]):
        bsoid_data[:,j] = conf[:,i]
        bsoid_data[:,j+1] = pos[:,i,0]
        bsoid_data[:,j+2] = pos[:,i,1]
        j += 3
    
    bodypart_headers = []
    for i in range(len(BSOID_DATA)):
        bodypart_headers.append(BSOID_DATA[i]+'_lh')
        bodypart_headers.append(BSOID_DATA[i]+'_x')
        bodypart_headers.append(BSOID_DATA[i]+'_y')
    
    bsoid_data = pd.DataFrame(bsoid_data)
    bsoid_data.columns = bodypart_headers

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
        if not os.path.exists(pose_est_dir + '/' + movie_name[0:-4] + "_pose_est_v2.h5"):
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
        else:
            logger.info(f'skipping {pose_est_dir}/{movie_name[0:-4]}_pose_est_v2.h5')