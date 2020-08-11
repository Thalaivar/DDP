import os
import glob
import math
import umap
import random
import joblib
import itertools
import hdbscan
import numpy as np
import pandas as pd

from tqdm import tqdm
from LOCAL_CONFIG import *
from preprocessing import *
from psutil import virtual_memory
from data import download_data, conv_bsoid_format
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier

def download(n):
    download_data('bsoid_strain_data.csv', BASE_PATH+RAW_DATA_DIR)
    
    print("Converting HDF5 files to csv files...")
    files = os.listdir(BASE_PATH+RAW_DATA_DIR)
    files = random.sample(files, n)
    for i in tqdm(range(len(files))):
        if files[i][-3:] == ".h5":
            conv_bsoid_format(BASE_PATH+RAW_DATA_DIR+files[i], BASE_PATH+CSV_DATA_DIR)

def process_csvs():
    csv_rep = glob.glob(BASE_PATH + CSV_DATA_DIR + '/*.csv')
    print('Preprocessing {} CSVs from {}'.format(len(csv_rep), BASE_PATH+CSV_DATA_DIR))
    curr_df = pd.read_csv(csv_rep[0], low_memory=False)
    currdf = np.array(curr_df)
    BP = np.unique(currdf[0,1:])
    BODYPARTS = []
    for b in BP:
        index = [i for i, s in enumerate(currdf[0, 1:]) if b in s]
        if not index in BODYPARTS:
            BODYPARTS += index
    BODYPARTS.sort()
    filenames = []
    rawdata_li = []
    data_li = []
    perc_rect_li = []
    f = get_filenames(BASE_PATH, CSV_DATA_DIR)
    for j, filename in enumerate(f):
        curr_df = pd.read_csv(filename, low_memory=False)
        curr_df_filt, perc_rect = adp_filt(curr_df, BODYPARTS)
        rawdata_li.append(curr_df)
        perc_rect_li.append(perc_rect)
        data_li.append(curr_df_filt)        
        filenames.append(filename)
    training_data = np.array(data_li)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_data.sav'))), 'wb') as f:
        joblib.dump([BASE_PATH, FPS, BODYPARTS, filenames, rawdata_li, training_data, perc_rect_li], f)

def process_feats():
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_data.sav'))), 'rb') as fr:
        BASE_PATH, FPS, BODYPARTS, filenames, rawdata_li, training_data, perc_rect_li = joblib.load(fr)
    win_len = np.int(np.round(0.05 / (1 / FPS)) * 2 - 1)
    feats = []
    print("Extracting features from {} CSV files..".format(len(training_data)))
    for m in tqdm(range(len(training_data))):
        dataRange = len(training_data[m])
        dxy_r = []
        dis_r = []
        for r in range(dataRange):
            if r < dataRange - 1:
                dis = []
                for c in range(0, training_data[m].shape[1], 2):
                    dis.append(np.linalg.norm(training_data[m][r + 1, c:c + 2] - training_data[m][r, c:c + 2]))
                dis_r.append(dis)
            dxy = []
            for i, j in itertools.combinations(range(0, training_data[m].shape[1], 2), 2):
                dxy.append(training_data[m][r, i:i + 2] - training_data[m][r, j:j + 2])
            dxy_r.append(dxy)
        dis_r = np.array(dis_r)
        dxy_r = np.array(dxy_r)
        dis_smth = []
        dxy_eu = np.zeros([dataRange, dxy_r.shape[1]])
        ang = np.zeros([dataRange - 1, dxy_r.shape[1]])
        dxy_smth = []
        ang_smth = []
        for l in range(dis_r.shape[1]):
            dis_smth.append(boxcar_center(dis_r[:, l], win_len))
        for k in range(dxy_r.shape[1]):
            for kk in range(dataRange):
                dxy_eu[kk, k] = np.linalg.norm(dxy_r[kk, k, :])
                if kk < dataRange - 1:
                    b_3d = np.hstack([dxy_r[kk + 1, k, :], 0])
                    a_3d = np.hstack([dxy_r[kk, k, :], 0])
                    c = np.cross(b_3d, a_3d)
                    ang[kk, k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                        math.atan2(np.linalg.norm(c),
                                                   np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :])))
            dxy_smth.append(boxcar_center(dxy_eu[:, k], win_len))
            ang_smth.append(boxcar_center(ang[:, k], win_len))
        dis_smth = np.array(dis_smth)
        dxy_smth = np.array(dxy_smth)
        ang_smth = np.array(ang_smth)
        feats.append(np.vstack((dxy_smth[:, 1:], ang_smth, dis_smth)))

    for n in range(0, len(feats)):
        feats1 = np.zeros(len(training_data[n]))
        for k in range(round(FPS / 10), len(feats[n][0]), round(FPS / 10)):
            if k > round(FPS / 10):
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((feats[n][0:dxy_smth.shape[0],
                                                             range(k - round(FPS / 10), k)]), axis=1),
                                                    np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                                            range(k - round(FPS / 10), k)]),
                                                           axis=1))).reshape(len(feats[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((np.mean((feats[n][0:dxy_smth.shape[0], range(k - round(FPS / 10), k)]), axis=1),
                                    np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                            range(k - round(FPS / 10), k)]), axis=1))).reshape(len(feats[0]), 1)
        if n > 0:
            f_10fps = np.concatenate((f_10fps, feats1), axis=1)
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T
            f_10fps_sc = np.concatenate((f_10fps_sc, feats1_sc), axis=1)
        else:
            f_10fps = feats1
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T
            f_10fps_sc = feats1_sc 
    
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'wb') as fr:
        joblib.dump([f_10fps, f_10fps_sc], fr)

def embedding(subsample=False):
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'rb') as fr:
        _, f_10fps_sc = joblib.load(fr)

    feats_train = f_10fps_sc.T
    del f_10fps_sc

    mem = virtual_memory()
    allowed_n = int((mem.available - 256000000)/(f_10fps_sc.shape[0]*32*100))
    print("Max points allowed due to memory: {} and data has point: {}".format(allowed_n, feats_train.shape[0]))

    if subsample and allowed_n < feats_train.shape[0]:
        print("Subsampling data to max allowable limit...")
        idx = np.random.permutation(np.arange(f_10fps_sc.shape[1]))[0:allowed_n]
        f_10fps = f_10fps[:, idx]
        f_10fps_sc = f_10fps_sc[:, idx]

    if mem.available > feats_train.shape[1] * feats_train.shape[0] * 32 * 100 + 256000000:
        trained_umap = umap.UMAP(n_neighbors=100, 
                                 **UMAP_PARAMS).fit(feats_train)
    else:
        print('Detecting that you are running low on available memory for this computation, setting low_memory so will take longer.')
        trained_umap = umap.UMAP(n_neighbors=100, low_memory=True,  # power law
                                 **UMAP_PARAMS).fit(feats_train)
    umap_embeddings = trained_umap.embedding_
    print(
        'Done non-linear transformation of **{}** instances from **{}** D into **{}** D.'.format(feats_train.shape[0],
                                                                                                 feats_train.shape[1],
                                                                                                 umap_embeddings.shape[
                                                                                                     1]))
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc = joblib.load(fr)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_umap.sav'))), 'wb') as f:
        joblib.dump([f_10fps, f_10fps_sc, umap_embeddings], f)

def check_mem():
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'rb') as fr:
        _, f_10fps_sc = joblib.load(fr)   
    
    mem = virtual_memory()
    allowed_n = int((mem.available - 256000000)/(f_10fps_sc.shape[0]*32*100))
    print("Max points allowed due to memory: {} and data has point: {}".format(allowed_n, f_10fps_sc.shape[1]))

def clustering():
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_umap.sav'))), 'rb') as f:
        _, _, umap_embeddings = joblib.load(f)

    cluster_range = [0.4, 1.2]
    print('Clustering with cluster size ranging from {}% to {}%'.format(cluster_range[0], cluster_range[1]))

    highest_numulab = -np.infty
    numulab = []
    min_cluster_range = np.linspace(cluster_range[0], cluster_range[1], 25)
    for min_c in min_cluster_range:
        trained_classifier = hdbscan.HDBSCAN(prediction_data=True,
                                                min_cluster_size=int(round(min_c * 0.01 * umap_embeddings.shape[0])),
                                                **HDBSCAN_PARAMS).fit(umap_embeddings)
        numulab.append(len(np.unique(trained_classifier.labels_)))
        if numulab[-1] > highest_numulab:
            logging.info('Adjusting minimum cluster size to maximize cluster number...')
            highest_numulab = numulab[-1]
            best_clf = trained_classifier
    assignments = best_clf.labels_
    soft_clusters = hdbscan.all_points_membership_vectors(best_clf)
    soft_assignments = np.argmax(soft_clusters, axis=1)
    logging.info('Done assigning labels for **{}** instances in **{}** D space'.format(*umap_embeddings.shape))
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters.sav'))), 'wb') as f:
        joblib.dump([assignments, soft_clusters, soft_assignments], f)

    print('Identified {} clusters...'.format(len(np.unique(soft_assignments))))

def classifier():
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_umap.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc, umap_embeddings = joblib.load(fr)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters.sav'))), 'rb') as fr:
        assignments, soft_clusters, soft_assignments = joblib.load(fr)
    feats_train, feats_test, labels_train, labels_test = train_test_split(f_10fps.T, soft_assignments.T,
                                                                          test_size=HLDOUT, random_state=23)

    print('Training feedforward neural network on randomly partitioned {}% of training data...'.format(
        (1 - HLDOUT) * 100))
    classifier = MLPClassifier(**MLP_PARAMS)
    classifier.fit(feats_train, labels_train)
    clf = MLPClassifier(**MLP_PARAMS)
    clf.fit(f_10fps.T, soft_assignments.T)
    nn_assignments = clf.predict(f_10fps.T)
    logging.info('Done training feedforward neural network '
            'mapping **{}** features to **{}** assignments.'.format(f_10fps.T.shape, soft_assignments.T.shape))
    scores = cross_val_score(classifier, feats_test, labels_test, cv=CV_IT, n_jobs=-1)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_neuralnet.sav'))), 'wb') as f:
        joblib.dump([feats_test, labels_test, classifier, clf, scores, nn_assignments], f)

if __name__ == "__main__":
    # process_csvs()
    # process_feats()
    embedding(subsample=True)