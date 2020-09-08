from bsoid import BSOID
import logging

logging.basicConfig(level=logging.DEBUG)
bsoid = BSOID('test', '/Users/dhruvlaad/IIT/DDP/data_custom')
bsoid.process_csvs()