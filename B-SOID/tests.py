import logging
logging.basicConfig(level=logging.INFO)

from sklearn.preprocessing import StandardScaler
from BSOID.features.bsoid_features import *

def format_data(data):
    fdata  = {}
    fdata['x'] = np.hstack((data['x'][:,:5], data['x'][:,6:]))
    fdata['y'] = np.hstack((data['y'][:,:5], data['y'][:,6:]))

    N, d = fdata['x'].shape
    data = np.zeros((N, 2*d))
    j = 0
    for i in range(d):
        data[:,j] = fdata['x'][:,i]
        data[:,j+1] = fdata['y'][:,i]
        j += 2
    
    return data

def process_feats(training_data, FPS):
    win_len = np.int(np.round(0.05 / (1 / FPS)) * 2 - 1)
    feats = []
    print("Extracting features from {} CSV files..".format(len(training_data)))
    for m in range(len(training_data)):
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
            dis_smth.append(smoothen_data(dis_r[:, l], win_len))
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
            dxy_smth.append(smoothen_data(dxy_eu[:, k], win_len))
            ang_smth.append(smoothen_data(ang[:, k], win_len))
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
            # f_10fps.append(feats1)

            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T

            f_10fps_sc = np.concatenate((f_10fps_sc, feats1_sc), axis=1)
            # f_10fps_sc.append(feats1_sc)
        else:
            f_10fps = feats1
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T
            f_10fps_sc = feats1_sc

    return f_10fps.T, f_10fps_sc.T

def main():
    from BSOID.bsoid import BSOID
    bsoid = BSOID.load_config('D:/IIT/DDP/data', 'bsoid_feats')
    fdata = bsoid.load_filtered_data()[23:25]
    format_fdata = [format_data(data) for data in fdata]

    hsu = process_feats(format_fdata, 30)

    me = [extract_feats(data, 30) for data in fdata]
    me = window_extracted_feats(me, 3)

    me_sc = [StandardScaler().fit_transform(data) for data in me]
    me = np.vstack(me)
    me_sc = np.vstack(me_sc)

    me = (me, me_sc)

    return hsu, me

if __name__ == "__main__":
    main()