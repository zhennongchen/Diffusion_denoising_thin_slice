import sys 
sys.path.append('/workspace/Documents')
import os
import pandas as pd
import numpy as np
import shutil
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list

main_path = '/mnt/camca_NAS/Portable_CT_data/IRB2022P002233_collected_202404'

# sheet = pd.read_excel(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static.xlsx'),dtype = {'Patient_ID': str, 'Patient_subID': str})
sheet = pd.read_excel(os.path.join('/mnt/camca_NAS/diffusion_ct_motion/examples/real_data/list/others/', 'NEW_CT_concise_collected_portable_motion_w_fixed_CT_info_Diffusion_results_blinded_reference_full_study.xlsx'), dtype={'Patient_ID': str, 'Patient_subID': str,'Patient_ID_fixed': str, 'Patient_subID_fixed': str})

for i in range(0,sheet.shape[0]):
    row = sheet.iloc[i]
    patient_id = row['Patient_ID']
    patient_subid = row['Patient_subID']
    print(f"Processing patient {patient_id} {patient_subid}...")

    original_folder = os.path.join(main_path, patient_id, patient_subid)

    assert os.path.exists(original_folder), f"Original folder {original_folder} does not exist."
    des_folder = os.path.join(main_path, 'need', patient_id, patient_subid)

    shutil.copytree(original_folder, des_folder, dirs_exist_ok=True)

    patient_id_fixed = row['Patient_ID_fixed']
    patient_subid_fixed = row['Patient_subID_fixed']
    print(f"Processing fixed patient {patient_id_fixed} {patient_subid_fixed}...")
    original_folder_fixed = os.path.join(main_path, patient_id_fixed, patient_subid_fixed)
    assert os.path.exists(original_folder_fixed), f"Original fixed folder {original_folder_fixed} does not exist."
    des_folder_fixed = os.path.join(main_path, 'need', patient_id_fixed, patient_subid_fixed)
    shutil.copytree(original_folder_fixed, des_folder_fixed, dirs_exist_ok=True)

