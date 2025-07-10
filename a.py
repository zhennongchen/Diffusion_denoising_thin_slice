import sys
sys.path.append('/workspace/Documents')

import argparse
import os
import sys
import subprocess
import nibabel as nb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Data_processing as Data_processing

build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 3)

avg_slice = False

results = []
for i in range(0,1):
    patient_id = patient_id_list[n[i]]
    patient_subid = patient_subid_list[n[i]]
    random_n = random_num_list[n[i]]
    print(patient_id, patient_subid, random_n)

    gt_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch73_1/gt_img.nii.gz')
    gt_img = nb.load(gt_file).get_fdata()
    # process gt
    shape = gt_img.shape
    gt_img_new = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2]-2))
    for i in range(1, gt_img.shape[2]-1):
        gt_img_new[:,:,i-1] = np.mean(gt_img[:,:,i-1:i+2], axis = 2)
    gt_img = np.copy(gt_img_new) if avg_slice else np.copy(gt_img)
    gt_img_brain = Data_processing.cutoff_intensity(gt_img, cutoff_low=-100, cutoff_high=100)
    print('done gt')

    condition_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch73_1/condition_img.nii.gz')
    condition_img = nb.load(condition_file).get_fdata() if avg_slice == False else nb.load(condition_file).get_fdata()[:,:,1:shape[2]-1]
    condition_img_brain = Data_processing.cutoff_intensity(condition_img, cutoff_low=-100, cutoff_high=100)
    print('done condition')

    supervised_file = os.path.join('/mnt/camca_NAS/denoising/models/supervised_DDPM_possion_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch46_1/pred_img.nii.gz')
    supervised_img = nb.load(supervised_file).get_fdata() if avg_slice == False else nb.load(supervised_file).get_fdata()[:,:,1:shape[2]-1]
    supervised_img_brain = Data_processing.cutoff_intensity(supervised_img, cutoff_low=-100, cutoff_high=100)
    print('done supervised')

    ddpm_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch73_1/pred_img.nii.gz')
    ddpm_img = nb.load(ddpm_file).get_fdata() if avg_slice == False else nb.load(ddpm_file).get_fdata()[:,:,1:shape[2]-1]
    ddpm_img_brain = Data_processing.cutoff_intensity(ddpm_img, cutoff_low=-100, cutoff_high=100)
    print('done ddpm')

    ddpm_avg_20_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch73avg/pred_img_scans20.nii.gz')
    ddpm_avg_20_img = nb.load(ddpm_avg_20_file).get_fdata() if avg_slice == False else nb.load(ddpm_avg_20_file).get_fdata()[:,:,1:shape[2]-1]
    ddpm_avg_20_img_brain = Data_processing.cutoff_intensity(ddpm_avg_20_img, cutoff_low=-100, cutoff_high=100)
    print('done ddpm_avg_20')

    # compare brain region
    mae_brain_motion, _, rmse_brain_motion, _, ssim_brain_motion,psnr_brain_motion = ff.compare(condition_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
    mae_brain_supervised, _, rmse_brain_supervised, _, ssim_brain_supervised,psnr_brain_supervised = ff.compare(supervised_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)

    mae_brain_ddpm, _, rmse_brain_ddpm, _, ssim_brain_ddpm,psnr_brain_ddpm = ff.compare(ddpm_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
    mae_brain_ddpm_avg_20, _, rmse_brain_ddpm_avg_20, _, ssim_brain_ddpm_avg_20,psnr_brain_ddpm_avg_20 = ff.compare(ddpm_avg_20_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)

    print('motion:', mae_brain_motion, rmse_brain_motion, ssim_brain_motion, psnr_brain_motion)
    print('mae_brain_supervised:', mae_brain_supervised, rmse_brain_supervised, ssim_brain_supervised, psnr_brain_supervised)
    print('ddpm:', mae_brain_ddpm, rmse_brain_ddpm, ssim_brain_ddpm, psnr_brain_ddpm)
    print('ddpm_avg_20:', mae_brain_ddpm_avg_20, rmse_brain_ddpm_avg_20, ssim_brain_ddpm_avg_20, psnr_brain_ddpm_avg_20)


affine = nb.load(gt_file).affine

gt_img_brain2 = np.copy(gt_img_brain)

gt_img_brain2[gt_img_brain <= 10] = 0
gt_img_brain2[gt_img_brain >= 90] = 0
gt_img_brain2[(gt_img_brain > 10) & (gt_img_brain < 90)] = 1

nb.save(nb.Nifti1Image(gt_img_brain2, affine), os.path.join('/mnt/camca_NAS/denoising/binary.nii.gz'))


# save gt_img_brain
nb.save(nb.Nifti1Image(gt_img_brain, affine), os.path.join('/mnt/camca_NAS/denoising/gt.nii.gz'))

# save condition_img_brain
nb.save(nb.Nifti1Image(condition_img_brain, affine), os.path.join('/mnt/camca_NAS/denoising/condition.nii.gz'))

# save supervised_img_brain
nb.save(nb.Nifti1Image(supervised_img_brain, affine), os.path.join('/mnt/camca_NAS/denoising/supervised.nii.gz'))
nb.save(nb.Nifti1Image(np.abs(supervised_img_brain-gt_img_brain), affine), os.path.join('/mnt/camca_NAS/denoising/supervised_diff.nii.gz'))

# save ddpm_img_brain
nb.save(nb.Nifti1Image(ddpm_img_brain, affine), os.path.join('/mnt/camca_NAS/denoising/ddpm.nii.gz'))

# save ddpm_avg_20_img_brain
nb.save(nb.Nifti1Image(ddpm_avg_20_img_brain, affine), os.path.join('/mnt/camca_NAS/denoising/ddpm_avg_20.nii.gz'))
nb.save(nb.Nifti1Image(np.abs(ddpm_avg_20_img_brain-gt_img_brain), affine), os.path.join('/mnt/camca_NAS/denoising/ddpm_avg_20_diff.nii.gz'))

