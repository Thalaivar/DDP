import os
import joblib
import ftplib
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)


class Animal:
    def __init__(self, **metadata):
        self.strain = metadata['strain']
        self.filename = metadata['network_filename']
        self.sex = metadata['sex']
        self.id = metadata['mouse_id']
        
    def extract_features(self):
        strain, data, movie_name = self.filename.split('/')
        
        session = ftplib.FTP("ftp.box.com")
        session.login("ae16b011@smail.iitm.ac.in", "Q0w9e8r7t6Y%Z")

        