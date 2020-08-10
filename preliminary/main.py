from bsoid.train import *
import bsoid.likelihoodprocessing as likelihoodprocessing
from data import download_data, conv_bsoid_format
import bsoid.train as bsoid_train
from bsoid.config.LOCAL_CONFIG import *
import ast

DATA_DIR = BASE_PATH + TRAIN_FOLDERS[0]

def download():
    download_data('bsoid_strain_data.csv', DATA_DIR)
    
    print("Converting HDF5 files to csv files...")
    files = os.listdir(DATA_DIR)
    for i in tqdm(range(len(files))):
        if files[i][-3:] == ".h5":
            conv_bsoid_format(DATA_DIR+files[i])

    print("Deleting HDF5 files...")
    for f in files:
        if f[-3:] == ".h5":
            os.remove(DATA_DIR+f)

def load_data(preprocess=True):
    # data preprocessing
    if preprocess:
        # preprocess data
        filenames, training_data, perc_rect = likelihoodprocessing.main(TRAIN_FOLDERS)
        # save preprocessed data
        with open(DATA_DIR+"filenames.txt", 'w') as f:
            for filename in filenames:
                f.write("%s\n"%filename)
        np.save(DATA_DIR+"perc_rect.npy", perc_rect)    
        np.save(DATA_DIR+"filtered_data.npy", training_data)
    
    else:
        print("Loading preprocessed data...")
        with open(DATA_DIR+"filenames.txt", 'r') as f:
            filenames = ast.literal_eval(f.read())
        # modify filenames for path to current system
        for i in range(len(filenames)):
            file_path = filenames[i].split('/')
            filenames[i] = DATA_DIR+file_path[-1]

        perc_rect = np.load(DATA_DIR+"perc_rect.npy", allow_pickle=True)
        training_data = np.load(DATA_DIR+"filtered_data.npy", allow_pickle=True)
    
    return filenames, training_data, perc_rect

def train(saved_feats=False):
    if saved_feats:
        data = np.load(DATA_DIR+"f_10fps.npy")
    else:
        data = np.load(DATA_DIR+"filtered_data.npy")
        
    # dimensionality reduction
    algo = "t-SNE"
    params = {"dimensions":2, "perplexity":np.sqrt(data.shape[1]), "theta":0.5}
    f_10fps, f_10fps_sc, embedding, scaler = bsoid_dim_reduce(data, algo=algo, saved_feats=saved_feats, params=params)
    
    if algo == "UMAP":
        filename = "embeddings_UMAP_" + str(params["dimensions"]) + "D_md" + str(params["min_dist"]) + "_nbr" + str(params["neighbors"]) + ".npy"
    elif algo == "t-SNE":
        filename = "embeddings_tSNE_" + str(params["dimensions"]) + "D_P" + str(params["perplexity"]) + "_th" + str(params["theta"]) + ".npy"
    np.save(BASE_PATH+"embed/"+filename, embedding)

    # # GMM training
    # for p in perplexity:
    #     params["perplexity"] = p

    #     if algo == "UMAP":
    #         filename = "embeddings_UMAP_" + str(params["dimensions"]) + "D_md" + str(params["min_dist"]) + "_nbr" + str(params["neighbors"]) + ".npy"
    #     elif algo == "t-SNE":
    #         filename = "embeddings_tSNE_" + str(params["dimensions"]) + "D_P" + str(params["perplexity"]) + "_th" + str(params["theta"]) + ".npy"
    #     embedding = np.load(BASE_PATH+"embed/"+filename)
        
    #     gmm_assignments = bsoid_gmm(embedding)
        
    #     if algo == "UMAP":
    #         filename = "assignments_UMAP_" + str(params["dimensions"]) + "D_md" + str(params["min_dist"]) + "_nbr" + str(params["neighbors"]) + ".npy"
    #     elif algo == "t-SNE":
    #         filename = "assignments_tSNE_" + str(params["dimensions"]) + "D_P" + str(params["perplexity"]) + "_th" + str(params["theta"]) + ".npy"
    #     np.save(BASE_PATH+"embed/"+filename, gmm_assignments)

    # SVC training
    # classifier, scores = bsoid_svm(f_10fps_sc, gmm_assignments)

    # return f_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores
    #return f_10fps, f_10fps_sc, trained_tsne, scaler

if __name__ == "__main__":
    # download all data
    # download()

    # load data
    # load_data(preprocess=True)

    # train the unsupervised classifier
    train(saved_feats=True)
