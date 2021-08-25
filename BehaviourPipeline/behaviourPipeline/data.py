import os
import h5py
import ftplib
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

from getpass import getpass

BSOID_DATA = ['NOSE', 'LEFT_EAR', 'RIGHT_EAR', 
        'BASE_NECK', 'FOREPAW1', 'FOREPAW2', 
        'CENTER_SPINE', 'HINDPAW1', 'HINDPAW2', 
        'BASE_TAIL', 'MID_TAIL', 'TIP_TAIL']

def process_h5py_data(f: h5py.File):
    data = list(f.keys())[0]
    keys = list(f[data].keys())

    conf = np.array(f[data][keys[0]])
    pos = np.array(f[data][keys[1]])

    f.close()
    
    return conf, pos

def extract_to_csv(filename, save_dir):
    f = h5py.File(filename, "r")
    # retain only filename
    filename = os.path.split(filename)[-1]
    
    conf, pos = process_h5py_data(f)

    bsoid_data = bsoid_format(conf, pos)
    bsoid_data.to_csv(save_dir + '/' + filename[:-3] +'.csv', index=False)
        
def bsoid_format(conf, pos):
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

def get_pose_data_dir(base_dir, network_filename):
    strain, data, movie_name = network_filename.split('/')

    strains = ["LL6-B2B", "LL5-B2B", "LL4-B2B", "LL3-B2B", "LL2-B2B", "LL1-B2B"]
    datasets = ["strain-survey-batch-2019-05-29-e/", "strain-survey-batch-2019-05-29-d/", "strain-survey-batch-2019-05-29-c/",
                "strain-survey-batch-2019-05-29-b/", "strain-survey-batch-2019-05-29-a/"]

    idx = strains.index(strain)

    data_dir = None
    if idx == 0:
        data_dir = base_dir + '/' + datasets[0] + strain + '/' + data
    elif idx == 5:
        data_dir = base_dir + '/' + datasets[4] + strain + '/' + data
    else:
        if os.path.exists(base_dir + datasets[idx-1] + strain + '/' + data):
            data_dir = base_dir + '/' + datasets[idx-1] + strain + '/' + data
        else:
            data_dir = base_dir + '/' + datasets[idx] + strain + '/' + data
    
    if data_dir is None:
        return None, None
        
    data_file = data_dir + movie_name[0:-4] + '_pose_est_v2.h5'
    return data_dir, data_file

def get_filename_in_dataset(data_dir, networkfilename):
    poses_dir, _ = get_pose_data_dir(data_dir, networkfilename)
    _, _, movie_name = networkfilename.split('/')
    filename = f'{poses_dir}/{movie_name[0:-4]}_pose_est_v2.h5'
    return filename
    
def push_folder_to_box(upload_dir, base_dir):
    session = ftplib.FTP("ftp.box.com")
    password = getpass("Box login password: ")
    session.login("ae16b011@smail.iitm.ac.in", password)

    upload_dir_name = upload_dir.split('/')[-1]

    master_dir = 'JAX-IITM Shared Folder/B-SOiD'
    session.cwd(f'{master_dir}/{base_dir}')

    # check if folder exists
    dir_exists, filelist = False, []
    session.retrlines('LIST', filelist.append)
    for f in filelist:
        if f.split()[-1] == upload_dir_name and f.upper().startswith('D'):
            dir_exists = True
    
    if not dir_exists:
        session.mkd(upload_dir_name)
    
    session.cwd(upload_dir_name)
    for f in os.listdir(upload_dir):
        upload_file = open(f'{upload_dir}/{f}', 'rb')
        session.storbinary(f'STOR {f}', upload_file)
    session.quit()

    print(f'Done uploading {upload_dir}')

def push_file_to_box(upload_file, base_dir):
    session = ftplib.FTP("ftp.box.com")
    
    password = getpass("Box login password: ")
    session.login("ae16b011@smail.iitm.ac.in", password)

    master_dir = 'JAX-IITM Shared Folder/B-SOiD'
    session.cwd(f'{master_dir}/{base_dir}')

    filename = upload_file.split('/')[-1]
    upload_file = open(upload_file, 'rb')
    session.storbinary(f'STOR {filename}', upload_file)
    session.quit()