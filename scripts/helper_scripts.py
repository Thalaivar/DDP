import os
import ftplib
import pandas as pd
from tqdm import tqdm
from getpass import getpass
from data import extract_to_csv
from preprocessing import likelihood_filter

def download_data_from_box(bsoid_data_file, output_dir, n=None):
    try: os.mkdir(output_dir)
    except FileExistsError: pass

    bsoid_data = pd.read_csv(bsoid_data_file)
    if n: bsoid_data = bsoid_data.sample(n).reset_index()

    session = ftplib.FTP("ftp.box.com")
    password = getpass("Box login password: ")
    session.login("ae16b011@smail.iitm.ac.in", password)

    strains = ["LL6-B2B", "LL5-B2B", "LL4-B2B", "LL3-B2B", "LL2-B2B", "LL1-B2B"]
    datasets = ["strain-survey-batch-2019-05-29-e/", "strain-survey-batch-2019-05-29-d/", "strain-survey-batch-2019-05-29-c/",
                "strain-survey-batch-2019-05-29-b/", "strain-survey-batch-2019-05-29-a/"]

    # master directory where datasets are saved
    master_dir = 'JAX-IITM Shared Folder/Datasets/'
    print(f"downloading data sets from box and saving output to {output_dir}")
    for i in tqdm(range(bsoid_data.shape[0])):
        strain, data, movie_name = bsoid_data['NetworkFilename'][i].split('/')
        if not os.path.exists(output_dir + '/' + movie_name[0:-4] + "_pose_est_v2.h5"):
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
            session.retrbinary("RETR "+ filename, open(output_dir + '/' + filename, 'wb').write)
            session.cwd('/')
        else:
            print(f'skipping {output_dir}/{movie_name[0:-4]}_pose_est_v2.h5')
    
    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".h5")]
    print("converting {} HDF5 files to csv files".format(len(files)))
    for f in files:
        extract_to_csv(f, output_dir)

def process_csvs(data_dir, bodyparts, fps, conf_threshold, filter_thresh):
    csv_data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    print('processing {} csv files from {}'.format(len(csv_data_files), data_dir))

    filtered_data = []
    skipped = 0
    for i, csv_file in enumerate(csv_data_files):
        data = pd.read_csv(csv_file)
        # fdata, perc_filt = likelihood_filter(data, fps, conf_threshold, bodyparts, end_trim, clip_window)
        fdata, perc_filt = likelihood_filter(data, conf_threshold, bodyparts)
        if fdata is not None and perc_filt < filter_thresh:
            assert fdata['x'].shape == fdata['y'].shape == fdata['conf'].shape, 'filtered data shape does not match across x, y, and conf values'
            filtered_data.append(fdata)
            shape = fdata['x'].shape
            print(f'preprocessed {shape} data from animal #{i}, with {round(perc_filt, 2)}% data filtered')
        else:
            print(f'skpping {i}-th dataset since % filtered is {round(perc_filt, 2)}')
            skipped += 1

    print(f'skipped {skipped}/{len(csv_data_files)} datasets')
    return filtered_data