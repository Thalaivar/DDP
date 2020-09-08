import os
import glob
import math
import umap
import random
import joblib
import itertools
import warnings
import ffmpeg
import hdbscan
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sn

from tqdm import tqdm
from LOCAL_CONFIG import *
from preprocessing import *
from psutil import virtual_memory
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from data import download_data, conv_bsoid_format
from utils import bsoid_extract, bsoid_predict
from videos import create_vids
from sklearn.model_selection import train_test_split, cross_val_score

def get_data(n=None, download=False):
    if download:
        download_data('bsoid_strain_data.csv', BASE_PATH+RAW_DATA_DIR)
    
    print("Converting HDF5 files to csv files...")
    files = os.listdir(BASE_PATH+RAW_DATA_DIR)
    if n is not None:
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

def embedding(dimensions=10, train_size=int(4e5)):
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc = joblib.load(fr)
    
    f_10fps = np.hstack(f_10fps)
    f_10fps_sc = np.hstack(f_10fps_sc)

    # train embedding map
    idx = np.random.permutation(np.arange(f_10fps.shape[1]))
    idx = idx[:train_size]
    feats_train = f_10fps_sc[:, idx]

    allowed_n = check_mem(feats_train)
    feats_train = feats_train.T

    if allowed_n > feats_train.shape[0]:
        print("Runnning UMAP on {} samples from {}D to {}D...".format(feats_train.shape[0], feats_train.shape[1], dimensions))
        mapper = umap.UMAP(n_components=dimensions, n_neighbors=100, **UMAP_PARAMS).fit(feats_train)
    else:
        raise MemoryError('max points allowed for UMAP {}, while you have {}'.format(allowed_n, feats_train.shape[1]))
    
    # embed entire dataset
    print("Embedding entire data set...")
    umap_embeddings = mapper.transform(f_10fps_sc.T)

    # save results
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_umap.sav'))), 'wb') as fr:
        joblib.dump([f_10fps, f_10fps_sc, umap_embeddings, mapper], fr)


def incremental_embedding(batch_sz=int(3e5), sliding_window=int(1e5), shuffle=True):
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc = joblib.load(fr)

    # collect feats from all animals into one array
    f_10fps = np.hstack(f_10fps)
    f_10fps_sc = np.hstack(f_10fps_sc)
    if shuffle:
        # shuffle features
        idx = np.random.permutation(np.arange(f_10fps.shape[1]))
        f_10fps, f_10fps_sc = f_10fps[:,idx], f_10fps_sc[:,idx]
    # no. of samples
    N  = f_10fps_sc.shape[1]

    print('Running incremental umap update on {} samples in batches of {}...'.format(N, batch_sz))
    
    # taken from https://gist.github.com/lmcinnes/a655c9382401a3e07915310f7611e8cf
    slices = []
    i = 0
    while sliding_window*i + batch_sz >= N:
        slices.append(f_10fps_sc[:, sliding_window*i:min(N, sliding_window*i + batch_sz)])
        i += 1
    relation_dict = {i+sliding_window:i for i in range(batch_sz-sliding_window)}
    relation_dicts = [relation_dict.copy() for i in range(len(slices))]

    aligned_mapper = umap.aligned_umap.AlignedUMAP(n_neighbors=100, **UMAP_PARAMS).fit(slices, relations=relation_dicts[:-1])
    embeddings = aligned_mapper.embeddings_

    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_umap.sav'))), 'wb') as fr:
        joblib.dump([f_10fps, f_10fps_sc, embeddings], fr)


def check_mem(f_10fps_sc):
    
    mem = virtual_memory()
    allowed_n = int((mem.available - 256000000)/(f_10fps_sc.shape[0]*32*100))
    print("Max points allowed due to memory: {} and data has point: {}".format(allowed_n, f_10fps_sc.shape[1]))

    return allowed_n

def clustering(umap_embeddings, cluster_range=None):
    if cluster_range is None:
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
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_umap_10D.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc, umap_embeddings = joblib.load(fr)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters_10D.sav'))), 'rb') as fr:
        assignments, soft_clusters, soft_assignments = joblib.load(fr)

    print("Training classifier on features...")
    # clf = MLPClassifier(**MLP_PARAMS)
    clf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
    clf.fit(f_10fps.T, soft_assignments.T)

    print('Done training feedforward neural network '
            'mapping **{}** features to **{}** assignments.'.format(f_10fps.T.shape, soft_assignments.T.shape))
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_neuralnet.sav'))), 'wb') as f:
        joblib.dump([clf], f)

def validate_classifier():
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_umap_10D.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc, umap_embeddings = joblib.load(fr)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters_10D.sav'))), 'rb') as fr:
        assignments, soft_clusters, soft_assignments = joblib.load(fr)
    
    print("Training and testing selected classifier on {}% paritioned data...".format((1 - HLDOUT) * 100))

    feats_train, feats_test, labels_train, labels_test = train_test_split(f_10fps.T, soft_assignments.T,
                                                                          test_size=HLDOUT, random_state=23)
    # classifier = MLPClassifier(**MLP_PARAMS)
    classifier = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
    classifier.fit(feats_train, labels_train)
    scores = cross_val_score(classifier, feats_test, labels_test, cv=CV_IT, n_jobs=-1)
    print("Classifier accuracy scores: {}".format(scores))

    labels_pred = classifier.predict(feats_test)
    data = confusion_matrix(labels_test, labels_pred, normalize='all')
    df_cm = pd.DataFrame(data, columns=np.unique(labels_test), index = np.unique(labels_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, cmap="Blues", annot=False)
    plt.show()

    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clf_results.sav'))), 'wb') as f:
        joblib.dump([feats_test, labels_test, labels_pred, scores, classifier], f)

    return df_cm, data, scores
    
def load_all():
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_neuralnet.sav'))), 'rb') as f:
        clf = joblib.load(f)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_umap.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc, umap_embeddings = joblib.load(fr)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters.sav'))), 'rb') as fr:
        assignments, soft_clusters, soft_assignments = joblib.load(fr)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clf_results.sav'))), 'rb') as f:
        df_cm, scores, classifier = joblib.load(f)
    
    return f_10fps, f_10fps_sc, umap_embeddings, assignments, soft_assignments, soft_clusters, scores, df_cm, classifier, clf

def results(csv_file, video_file, extract_frames=True):
    output_dir = TEST_DIR + csv_file.split('/')[-1][:-4]
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    frame_dir = output_dir + '/pngs'
    try:
        os.mkdir(frame_dir)
    except FileExistsError:
        pass
    print('Extracting frames from video {} to dir {}...'.format(video_file, frame_dir))
    
    probe = ffmpeg.probe(video_file)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])
    bit_rate = int(video_info['bit_rate'])
    avg_frame_rate = round(int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(video_info['avg_frame_rate'].rpartition('/')[2]))
    if extract_frames:
        print('Frame extraction for {} frames at {} frames per second'.format(num_frames, avg_frame_rate))
        try:
            (ffmpeg.input(video_file)
                .filter('fps', fps=avg_frame_rate)
                .output(str.join('', (frame_dir, '/frame%01d.png')), video_bitrate=bit_rate,
                        s=str.join('', (str(int(width * 0.5)), 'x', str(int(height * 0.5)))), sws_flags='bilinear',
                        start_number=0)
                .run(capture_stdout=True, capture_stderr=True))
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
    
    try:
        os.mkdir(output_dir + '/mp4s')
    except FileExistsError:
        pass

    shortvid_dir = output_dir + '/mp4s'
    print('Saving example videos from {} to {}...'.format(video_file, shortvid_dir))
    
    min_time = input('Minimum bout duration: ')
    min_frames = round(float(min_time) * 0.001 * float(FPS))
    
    number_examples = int(input('Number of examples per group: '))
    out_fps = int(input('Output frame rate: '))
    
    load_feats = input('Load pre-extracted features (yes/no): ')

    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_neuralnet.sav'))), 'rb') as fr:
        clf = joblib.load(fr)
    
    clf = clf[0]
    curr_df = pd.read_csv(csv_file, low_memory=False) 
    currdf = np.array(curr_df)
    BP = np.unique(currdf[0,1:])
    BODYPARTS = []
    for b in BP:
        index = [i for i, s in enumerate(currdf[0, 1:]) if b in s]
        if not index in BODYPARTS:
            BODYPARTS += index
    BODYPARTS.sort()
    
    curr_df_filt, perc_rect = adp_filt(curr_df, BODYPARTS)
    print('%% data below threshold: {}'.format(np.max(np.array(perc_rect))))
    test_data = [curr_df_filt]
    labels_fs = []
    labels_fs2 = []
    fs_labels = []
    for i in range(0, len(test_data)):
        if load_feats == "no":
            print('Extracting features from csv file {}...'.format(csv_file))
            feats_new = bsoid_extract(test_data, FPS)
            with open(OUTPUT_PATH + '/' + csv_file.split('/')[-1][:-4] + '_predict_feats.sav', 'wb') as fr:
                joblib.dump(feats_new, fr)
        else:
            print('Loading features from csv file {}...'.format(csv_file))
            with open(OUTPUT_PATH + '/' + csv_file.split('/')[-1][:-4] + '_predict_feats.sav', 'rb') as fr:
                feats_new = joblib.load(fr)

        labels = bsoid_predict(feats_new, clf)
        for m in range(0, len(labels)):
            labels[m] = labels[m][::-1]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        labels_fs.append(labels_pad.astype(int))
    for k in range(0, len(labels_fs)):
        labels_fs2 = []
        for l in range(math.floor(FPS / 10)):
            labels_fs2.append(labels_fs[k][l])
        fs_labels.append(np.array(labels_fs2).flatten('F'))
    # create_labeled_vid(fs_labels[0], int(min_frames), int(number_examples), int(out_fps),
    #                     frame_dir, shortvid_dir)
    create_vids(fs_labels[0], int(min_frames), int(number_examples), int(out_fps),
                          frame_dir, shortvid_dir)

    save_output = input('Save prediction output (yes/no): ')
    if save_output == "yes":
        with open(output_dir+'/'+MODEL_NAME+'_predict_data.sav', 'wb') as fr:
            joblib.dump([fs_labels, min_frames, out_fps], fr)

def load_predict_outputs():
    with open(output_dir+'/'+MODEL_NAME+'_predict_data.sav', 'rb') as fr:
        fs_labels, min_frames, out_fps = joblib.load(fr)
    return fs_labels, min_frames, out_fps
    
if __name__ == "__main__":
    process_csvs()
    # process_feats()
    # embedding()
    # incremental_embedding(8)
    # CURE(10)
    # clustering()
    # valdiate_classifier()
    # classifier()