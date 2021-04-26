import yaml
import pandas as pd

from BSOID.preprocessing import likelihood_filter
from BSOID.data import get_pose_data_dir

def load_from_dataset(config, n=None, n_strains=None):
    input_csv = config["JAX_DATASET"]["input_csv"]
    data_dir = config["JAX_DATASET"]["data_dir"]
    filter_thresh = config["filter_thresh"]

    if input_csv.endswith('.tsv'):
        all_data = pd.read_csv(input_csv, sep='\t')    
    else:
        all_data = pd.read_csv(input_csv)
    
    all_data = list(all_data.groupby("Strain"))
    random.shuffle(all_data)
    
    if n_strains is None:
        n_strains = len(all_data)
    
    strain_count, filtered_data = 0, {}
    for i in range(len(all_data)):
        if strain_count > n_strains:
            break

        n = all_data[i][1].shape[0] if n is None else n
        shuffled_strain_data = all_data[i][1].sample(frac=1)
        count, strain_fdata = 0, []
        for j in range(shuffled_strain_data.shape[0]):
            if count > n:
                break

            metadata = dict(shuffled_strain_data.iloc[j])
            try:
                pose_dir, _ = get_pose_data_dir(data_dir, metadata['NetworkFilename'])
                _, _, movie_name = metadata['NetworkFilename'].split('/')
                filename = f'{pose_dir}/{movie_name[0:-4]}_pose_est_v2.h5'

                f = h5py.File(filename, "r")
                filename = filename.split('/')[-1]
                data = list(f.keys())[0]
                keys = list(f[data].keys())
                conf, pos = np.array(f[data][keys[0]]), np.array(f[data][keys[1]])
                f.close()

                bsoid_data = bsoid_format(conf, pos)
                fdata, perc_filt = likelihood_filter(bsoid_data, self.fps, self.conf_threshold, **self.trim_params)
                strain, mouse_id = metadata['Strain'], metadata['MouseID']
                
                if perc_filt > filter_thresh:
                    logging.warning(f'mouse:{strain}/{mouse_id}: % data filtered from raw data is too high ({perc_filt} %)')
                else:
                    shape = fdata['x'].shape
                    logging.debug(f'preprocessed {shape} data from {strain}/{mouse_id} with {round(perc_filt, 2)}% data filtered')
                    strain_fdata.append(fdata)
                    count += 1

            except:
                pass
        
        if count - 1 == n:
            filtered_data[all_data[i][0]] = strain_fdata
            logging.info(f"extracted {count - 1} animal data for strain {all_data[i][0]}")
            strain_count += 1
        
    logging.info(f"extracted data from {strain_count - 1} strains with a total of {len(filtered_data)} animals")
    with open(self.output_dir + '/' + self.run_id + '_filtered_data.sav', 'wb') as f:
        joblib.dump(filtered_data, f)

    return filtered_data

def features_from_points(filtered_data, parallel=False):
    filtered_data = self.load_filtered_data()
    logging.info(f'extracting features from {len(filtered_data)} animals')

    # extract geometric features
    if parallel:
        feats = Parallel(n_jobs=-1)(delayed(extract_feats)(data, self.fps, self.stride_window) for data in filtered_data)
    else:
        feats = []
        for i in tqdm(range(len(filtered_data))):
            feats.append(extract_feats(filtered_data[i], self.fps, self.stride_window))

    logging.info(f'extracted {len(feats)} datasets of {feats[0].shape[1]}D features')

    # feats = window_extracted_feats(feats, self.stride_window, self.temporal_window, self.temporal_dims)
    feats = window_extracted_feats(feats, self.stride_window)
    logging.info(f'collected features into bins of {1000 * self.stride_window // self.fps} ms')

    with open(self.output_dir + '/' + self.run_id + '_features.sav', 'wb') as f:
        joblib.dump(feats, f)