#!/usr/bin/env python

import os
import sys
import shutil
import requests

__author__      = "Reza"
__copyright__   = "Copyright 2018/19, CAMP-MLMI Project, TUM"
__email__       = "reza.rahman@tum.de"
__maintainer__  = "Reza"
__status__      = "Dev"

## Constant Variables

CHUNK_SIZE = 32768
URL = "https://docs.google.com/uc?export=download"
FILE_NAME_1 = 'dim_reduction_output.txt'
FILE_NAME_2 = 'correlation_matrix.txt'
FILE_NAME_3 = 'ppmi_labels.csv'
FILE_NAME_4 = 'non_imaging_ppmi_data.xls'

patient_file_link_id = '1KpZ_vGsD_BvXJBUhEtGfYLcvDANv401v'
dim_reduction_file_link_id = '1xlEljkeG0TkxLsd1fkZ4diNhZyvnll07'
correlation_matrix_file_link_id = '1Z7hfncKKbvYJSO22uYvyMzQ68CqYawpe'
non_imaging_ppmi_data_link_id = '1aDx_tNli1FqfU4L3yWcfoNIEAgVreBW_'

dataset_folder_name = 'Dataset'


class DatasetDownload():
    def get_confirmation_token(self, res):
        for key, value in res.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    
    def save_file_content(self, res, file_name):
        response_string = str(res)
        print(response_string)
        if response_string == '<Response [200]>':
            with open(file_name, "wb") as file:
                for chunk in res.iter_content(CHUNK_SIZE):
                    if chunk: # filter out keep-alive new chunks
                        file.write(chunk)
                        
    def save_file_location(self):
        # Creating Dataset Folder
        if not os.path.exists(dataset_folder_name):
            os.makedirs(dataset_folder_name)
        
        shutil.move(FILE_NAME_1, os.path.join(dataset_folder_name))
        shutil.move(FILE_NAME_2, os.path.join(dataset_folder_name))
        shutil.move(FILE_NAME_3, os.path.join(dataset_folder_name))
        shutil.move(FILE_NAME_4, os.path.join(dataset_folder_name))

        print("Dataset Successfully downloaded! ... ")
        
    def start_session(self):
        sess = requests.Session()
        
        # Get responses for ""to be" downloaded files
        response_patients = sess.get(URL, params = { 'id' : patient_file_link_id }, stream = True)
        response_dim_reduc = sess.get(URL, params = { 'id' : dim_reduction_file_link_id }, stream = True)
        response_cor_mat = sess.get(URL, params = { 'id' : correlation_matrix_file_link_id }, stream = True)
        response_ppmi_data = sess.get(URL, params = { 'id' : non_imaging_ppmi_data_link_id }, stream = True)

        #token = self.get_confirmation_token(response)
        #print("token is : ", token)

        '''if token:
            params = { 'id' : file_id, 'confirm' : token }
            response = sess.get(URL, params = params, stream = True) '''
        
        # Based on response, download the files
        self.save_file_content(response_patients, FILE_NAME_3)
        self.save_file_content(response_dim_reduc, FILE_NAME_1)
        self.save_file_content(response_cor_mat, FILE_NAME_2)
        self.save_file_content(response_ppmi_data, FILE_NAME_4)
        
if __name__ == "__main__":
    downloader_obj = DatasetDownload()
    downloader_obj.start_session() 
    downloader_obj.save_file_location()
