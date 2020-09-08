import os
import joblib
import itertools
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from LOCAL_CONFIG import *
from preprocessing import boxcar_center

def process_feats():
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_data.sav'))), 'rb') as fr:
        BASE_PATH, FPS, BODYPARTS, filenames, rawdata_li, training_data, perc_rect_li = joblib.load(fr)
    win_len = np.int(np.round(0.05 / (1 / FPS)) * 2 - 1)
    
    pool = mp.Pool(mp.cpu_count())
    
    feats = []
    print("Extracting features from {} CSV files..".format(len(training_data)))
    for m in tqdm(range(len(training_data))):
        dataRange = len(training_data[m])

        dxy_r = pool.starmap(dxy, [(r, dataRange, training_data[m]) for r in range(dataRange)])
        dis_r = pool.starmap(dis, [(r, dataRange, training_data[m]) for r in range(dataRange)])
        dis_r = np.array(dis_r)
        dxy_r = np.array(dxy_r)

        dis_smth = []
        dxy_smth = []
        ang_smth = []
        for l in range(dis_r.shape[1]):
            dis_smth.append(boxcar_center(dis_r[:, l], win_len))
        
        for k in range(dxy_r.shape[1]):
            
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
            # f_10fps = np.concatenate((f_10fps, feats1), axis=1)
            f_10fps.append(feats1)

            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T

            # f_10fps_sc = np.concatenate((f_10fps_sc, feats1_sc), axis=1)
            f_10fps_sc.append(feats1_sc)
        else:
            f_10fps = [feats1]
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T
            f_10fps_sc = [feats1_sc]
    
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'wb') as fr:
        joblib.dump([f_10fps, f_10fps_sc], fr)

# split into functions to enable parallelism
def dxy(r, dataRange, training_data):
    dxy = []
    for i, j in itertools.combinations(range(0, training_data.shape[1], 2), 2):
        dxy.append(training_data[r, i:i + 2] - training_data[r, j:j + 2])
    
    return dxy

def dis(r, dataRange, training_data):
    if r < dataRange - 1:
        dis = []
        for c in range(0, training_data.shape[1], 2):
            dis.append(np.linalg.norm(training_data[r + 1, c:c + 2] - training_data[r, c:c + 2]))
    
    return dis

def dxy_smth(k, dataRange, dxy_r, win_len):
    dxy_eu = []
    for kk in range(dataRange):
        dxy_eu.append(np.linalg.norm(dxy_r[kk, k, :]))
    
    dxy_eu = np.array(dxy_eu)
    return boxcar_center(dxy_eu, win_len)

def ang_smth(k, dataRange, dxy_r, win_len):
    ang = []
    if kk < dataRange - 1:
        b_3d = np.hstack([dxy_r[kk + 1, k, :], 0])
        a_3d = np.hstack([dxy_r[kk, k, :], 0])
        c = np.cross(b_3d, a_3d)
        ang.append(np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                            math.atan2(np.linalg.norm(c),
                                        np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :]))))

    ang = np.array(ang)
    return boxcar_center(ang, win_len)