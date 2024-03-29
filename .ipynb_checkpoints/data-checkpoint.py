import os
import h5py
import ftplib
import random

import pandas as pd
import numpy as np

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

from getpass import getpass

BSOID_DATA = ['NOSE', 'LEFT_EAR', 'RIGHT_EAR', 
        'BASE_NECK', 'FOREPAW1', 'FOREPAW2', 
        'CENTER_SPINE', 'HINDPAW1', 'HINDPAW2', 
        'BASE_TAIL', 'MID_TAIL', 'TIP_TAIL']

RETAIN_WINDOW = 30*60
FPS = 30

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

def download_data(bsoid_data_file, pose_est_dir):
    bsoid_data = pd.read_csv(bsoid_data_file)
    
    session = ftplib.FTP("ftp.box.com")
    password = getpass("Box login password: ")
    session.login("ae16b011@smail.iitm.ac.in", password)

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
    
def download_raw_data_files(n: int):
        os.mkdir(os.path.join(self.base_dir, "tmp"))
        download_data('bsoid_strain_data.csv', self.raw_dir)
        
        files = os.listdir(self.raw_dir)
        logger.info("converting {} HDF5 files to csv files".format(len(files)))
        if n is not None:
            files = random.sample(files, n)
        if parallel:
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1)(delayed(extract_to_csv)(self.raw_dir+'/'+files[i], self.csv_dir) for i in range(len(files)))
        else:
            for i in tqdm(range(len(files))):
                if files[i][-3:] == ".h5":
                    extract_to_csv(self.raw_dir+'/'+files[i], self.csv_dir)