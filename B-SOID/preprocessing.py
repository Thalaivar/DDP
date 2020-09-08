import pandas as pd
import numpy as np
import logging
import itertools

def smoothen_data(data, win_len=7):
    data = pd.Series(data)
    smoothed_data = data.rolling(win_len, min_periods=1, center=True)
    return np.array(smoothed_data.mean())

def likelihood_filter(data: pd.DataFrame, conf_threshold: float=0.3, forward_fill=True):
    N = data.shape[0]
    n_dpoints = data.shape[1] // 3
    # retrieve confidence, x and y data from csv data
    conf, x, y = [], [], []
    for col in data.columns:
        if col.endswith('_lh'):
            conf.append(data[col])
        elif col.endswith('_x'):
            x.append(data[col])
        elif col.endswith('_y'):
            y.append(data[col])
    conf, x, y = np.array(conf).T, np.array(x).T, np.array(y).T

    logging.info('extracted {} samples of {} features'.format(N, n_dpoints))

    # forward-fill any points below confidence threshold
    if forward_fill:
        perc_filt = []
        for i in range(n_dpoints):
            n_filtered = 0
            if conf[0,i] < conf_threshold:
                # find first good confidence point
                k = 0
                while conf[k,i] < conf_threshold:
                    n_filtered += 1
                    k += 1
                # replace all points with first good conf point
                conf[0:k,i] = conf[k,i]*np.ones_like(conf[0:k,i])
                x[0:k,i] = x[k,i]*np.ones_like(x[0:k,i])
                y[0:k,i] = y[k,i]*np.ones_like(y[0:k,i])
                
                prev_lh_idx = k
            else:
                prev_lh_idx = 0                
                k = 0

            for j in range(k, N):
                if conf[j,i] < conf_threshold:
                    # if current point is low confidence, replace with last confident point
                    x[j,i], y[j,i] = x[prev_lh_idx,i], y[prev_lh_idx,i]
                    n_filtered += 1
                else:
                    prev_lh_idx = j
        
            perc_filt.append(n_filtered)
        perc_filt = [(p/N)*100 for p in perc_filt]
        logging.info('%% filtered from all features: {}'.format(perc_filt))
    
    return {'conf': conf, 'x': x, 'y': y}

# calculate required features from x, y position data
def feats_from_xy(x, y):
    # indices -> features
    HEAD, BASE_NECK, CENTER_SPINE, HINDPAW1, HINDPAW2, BASE_TAIL, MID_TAIL, TIP_TAIL = np.arange(8)


    # link connections [start, end]
    link_connections = ([BASE_NECK, HEAD], 
                    [CENTER_SPINE, BASE_NECK],
                    [BASE_TAIL, CENTER_SPINE], 
                    [BASE_TAIL, HINDPAW1], [BASE_TAIL, HINDPAW2],
                    [BASE_TAIL, MID_TAIL],
                    [MID_TAIL, TIP_TAIL])

    # links between body points
    links = []
    for conn in link_connections:
        links.append([x[conn[0]] - x[conn[1]], y[conn[0]] - y[conn[1]]])
    links = np.array(links)

    # relative angles between links
    angles = []
    

def extract_bsoid_feats(filtered_data):
    x, y = filtered_data['x'], filtered_data['y']
    N, n_dpoints = x.shape

    # displacements of all points
    dis = []
    for i in range(N - 1):
        dis_vec = np.array([x[i+1,:] - x[i,:], y[i+1,:] - y[i,:]])
        dis.append(np.linalg.norm(dis_vec, axis=0))
    dis = np.array(dis)

    # link lengths of all possible combinations
    links = []
    for k in range(N):
        curr_links = []
        for i, j in itertools.combinations(range(n_dpoints), 2):
            curr_links.append([x[k,i] - x[k,j], y[k,i] - y[k,j]])
        links.append(curr_links)
    links = np.array(links)

    # angles between link position for two timesteps
    angles = []
    for i in range(N-1):
        curr_angles = []
        for j in range(links.shape[1]):
            link_dis_cross = np.cross(links[i,j], links[i+1,j])[0]            
            curr_angles.append(math.atan2(link_dis_cross, links[i,j].dot(links[i+1,j])))
        angles.append(curr_angles)
    
    