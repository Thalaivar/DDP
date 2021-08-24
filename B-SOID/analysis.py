from BSOID.analysis import *
from BSOID.bsoid import *

import joblib
with open("D:/IIT/DDP/data/finals/label_info.sav", "rb") as f:
    label_info = joblib.load(f)

bsoid = BSOID("D:/IIT/DDP/DDP/B-SOID/config/config.yaml")
input_csv = pd.read_csv("D:/IIT/DDP/data/finals/StrainSurveyMetaList_2019-04-09.tsv", sep='\t')
gemma_files(label_info, input_csv, max_label=61, min_bout_len=200, fps=30, default_config_file="D:/IIT/DDP/DDP/B-SOID/gemma/default_GEMMA.yaml")