import sys 
sys.path.append('/workspace/Documents')
import os
import pandas as pd
import numpy as np
import shutil
import Diffusion_denoising_thin_slice.functions_collection as ff

### transfer data from NAS to local
nas_path = '/mnt/camca_NAS/denoising/Data'
local_path = '/workspace/Documents/Data/denoising'

patient_sheet = pd.read_excel(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_shuffled_batched.xlsx'),dtype={'Patient_ID': str, 'Patient_subID': str})
noise_types = ['gaussian','possion']
simulation_num = 2

for noise_type in noise_types:
    results = []
    for i in range(0, len(patient_sheet)):
        patient_id = patient_sheet['Patient_ID'][i]
        patient_subid = patient_sheet['Patient_subID'][i]
        batch = patient_sheet['batch'][i]
        print(f"Processing patient {patient_id} {patient_subid}...")

        ground_truth_file = os.path.join(nas_path,'fixedCT',patient_id,patient_subid,'img_thinslice.nii.gz')
        ff.make_folder([os.path.join(local_path, 'fixedCT'), os.path.join(local_path, 'fixedCT',patient_id),os.path.join(local_path, 'fixedCT',patient_id,patient_subid)])
        if not os.path.exists(os.path.join(local_path,'fixedCT',patient_id,patient_subid,'img_thinslice.nii.gz')):
            shutil.copy(ground_truth_file,os.path.join(local_path,'fixedCT',patient_id,patient_subid,'img_thinslice.nii.gz'))

        for simulation_n in range(2,3):
            simulation_files = os.path.join(nas_path,'simulation',patient_id,patient_subid,noise_type + '_random_' + str(simulation_n), 'recon.nii.gz')
            ff.make_folder([os.path.join(local_path, 'simulation'), os.path.join(local_path, 'simulation',patient_id),os.path.join(local_path, 'simulation',patient_id,patient_subid),os.path.join(local_path, 'simulation',patient_id,patient_subid,noise_type + '_random_' + str(simulation_n))])
            if not os.path.exists(os.path.join(local_path,'simulation',patient_id,patient_subid,noise_type + '_random_' + str(simulation_n),'recon.nii.gz')):
                shutil.copy(simulation_files,os.path.join(local_path,'simulation',patient_id,patient_subid,noise_type + '_random_' + str(simulation_n),'recon.nii.gz'))
