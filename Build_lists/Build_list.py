import numpy as np
import os
import pandas as pd


class Build():
    def __init__(self,file_list):
        self.a = 1
        self.file_list = file_list
        # self.data = pd.read_excel(file_list, dtype = {'Patient_ID': str, 'Patient_subID': str})
        self.data = pd.read_excel(file_list, dtype = {'Patient_ID': str})

    def __build__(self,batch_list):
        for b in range(len(batch_list)):
            cases = self.data.loc[self.data['batch'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])

        batch_list = np.asarray(c['batch'])
        patient_id_list = np.asarray(c['Patient_ID'])
        random_num_list = np.asarray(c['random_num'])
        noise_file_all_list = np.asarray(c['simulation_file_all']) 
        noise_file_odd_list = np.asarray(c['simulation_file_odd'])
        noise_file_even_list = np.asarray(c['simulation_file_even'])
        ground_truth_file_list = np.asarray(c['ground_truth_file']) 
        slice_num_list = np.asarray(c['slice_num']) if 'slice_num' in c.columns else None

        
        return batch_list, patient_id_list, random_num_list, noise_file_all_list, noise_file_odd_list, noise_file_even_list, ground_truth_file_list, slice_num_list
