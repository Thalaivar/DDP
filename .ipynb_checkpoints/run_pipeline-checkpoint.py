from behaviourPipeline.pipeline import BehaviourPipeline

import logging
logging.basicConfig(level=logging.INFO)

import argparse
import pandas as pd
parser = argparse.ArgumentParser("run_pipeline.py")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--n-strains", type=int, default=-1)
parser.add_argument("--records", type=str, required=True)
parser.add_argument("--data-dir", type=str, required=True)

args = parser.parse_args()
pipeline = BehaviourPipeline(args.name, args.config)

records = pd.read_csv(args.records, sep="\t")
pipeline.ingest_data(args.data_dir, records, args.n, args.n_strains)
pipeline.compute_features()
pipeline.cluster_strainwise()
pipeline.pool()
pipeline.train()