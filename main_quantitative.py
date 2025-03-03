import sys
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np 
import nibabel as nb
import pandas as pd
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Data_processing as Data_processing

build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 

results = []
for i in list(range(0, 3+ 1, 3)):#range(0,1):#x0_list.shape[0]):
    patient_id = patient_id_list[i]
    patient_subid = patient_subid_list[i]
    random_n = random_num_list[i]
    print(patient_id, patient_subid, random_n)

    gt_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch70_1/gt_img.nii.gz')
    gt_img = nb.load(gt_file).get_fdata()
    gt_img_brain = Data_processing.cutoff_intensity(gt_img, cutoff_low=-100, cutoff_high=100)

    condition_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch70_1/condition_img.nii.gz')
    condition_img = nb.load(condition_file).get_fdata()
    condition_img_brain = Data_processing.cutoff_intensity(condition_img, cutoff_low=-100, cutoff_high=100)

    noise2noise_file = os.path.join('/mnt/camca_NAS/denoising/models/noise2noise_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch77/pred_img.nii.gz')
    noise2noise_img = nb.load(noise2noise_file).get_fdata()
    noise2noise_img_brain = Data_processing.cutoff_intensity(noise2noise_img, cutoff_low=-100, cutoff_high=100)

    ddpm_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch70_1/pred_img.nii.gz')
    ddpm_img = nb.load(ddpm_file).get_fdata()
    ddpm_img_brain = Data_processing.cutoff_intensity(ddpm_img, cutoff_low=-100, cutoff_high=100)

    ddpm_avg_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch70final/pred_img.nii.gz')
    ddpm_avg_img = nb.load(ddpm_avg_file).get_fdata()
    ddpm_avg_img_brain = Data_processing.cutoff_intensity(ddpm_avg_img, cutoff_low=-100, cutoff_high=100)

    # compare brain region
    mae_brain_motion, _, rmse_brain_motion, _, ssim_brain_motion,psnr_brain_motion = ff.compare(condition_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
    mae_brain_n2n, _, rmse_brain_n2n, _, ssim_brain_n2n,psnr_brain_n2n = ff.compare(noise2noise_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
    mae_brain_ddpm, _, rmse_brain_ddpm, _, ssim_brain_ddpm,psnr_brain_ddpm = ff.compare(ddpm_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
    mae_brain_ddpm_avg, _, rmse_brain_ddpm_avg, _, ssim_brain_ddpm_avg,psnr_brain_ddpm_avg = ff.compare(ddpm_avg_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)

    print('motion:', mae_brain_motion, rmse_brain_motion, ssim_brain_motion, psnr_brain_motion)
    print('n2n:', mae_brain_n2n, rmse_brain_n2n, ssim_brain_n2n, psnr_brain_n2n)
    print('ddpm:', mae_brain_ddpm, rmse_brain_ddpm, ssim_brain_ddpm, psnr_brain_ddpm)
    print('ddpm_avg:', mae_brain_ddpm_avg, rmse_brain_ddpm_avg, ssim_brain_ddpm_avg, psnr_brain_ddpm_avg)

    # compare all 
    mae_motion, _, rmse_motion, _, ssim_motion,psnr_motion = ff.compare(condition_img, gt_img, cutoff_low = -100)
    mae_n2n, _, rmse_n2n, _, ssim_n2n,psnr_n2n = ff.compare(noise2noise_img, gt_img, cutoff_low = -100)
    mae_ddpm, _, rmse_ddpm, _, ssim_ddpm,psnr_ddpm = ff.compare(ddpm_img, gt_img, cutoff_low = -100)
    mae_ddpm_avg, _, rmse_ddpm_avg, _, ssim_ddpm_avg,psnr_ddpm_avg = ff.compare(ddpm_avg_img, gt_img, cutoff_low = -100)

    print('all image:')
    print('motion:', mae_motion, rmse_motion, ssim_motion, psnr_motion)
    print('n2n:', mae_n2n, rmse_n2n, ssim_n2n, psnr_n2n)
    print('ddpm:', mae_ddpm, rmse_ddpm, ssim_ddpm, psnr_ddpm)
    print('ddpm_avg:', mae_ddpm_avg, rmse_ddpm_avg, ssim_ddpm_avg, psnr_ddpm_avg)

    results.append([patient_id, patient_subid, random_n, mae_brain_motion, rmse_brain_motion, ssim_brain_motion, psnr_brain_motion, mae_brain_n2n, rmse_brain_n2n, ssim_brain_n2n, psnr_brain_n2n, mae_brain_ddpm, rmse_brain_ddpm, ssim_brain_ddpm, psnr_brain_ddpm, mae_brain_ddpm_avg, rmse_brain_ddpm_avg, ssim_brain_ddpm_avg, psnr_brain_ddpm_avg, mae_motion, rmse_motion, ssim_motion, psnr_motion, mae_n2n, rmse_n2n, ssim_n2n, psnr_n2n, mae_ddpm, rmse_ddpm, ssim_ddpm, psnr_ddpm, mae_ddpm_avg, rmse_ddpm_avg, ssim_ddpm_avg, psnr_ddpm_avg])

    df = pd.DataFrame(results , columns = ['patient_id', 'patient_subid', 'random_num', 'mae_brain_motion', 'rmse_brain_motion', 'ssim_brain_motion', 'psnr_brain_motion', 'mae_brain_n2n', 'rmse_brain_n2n', 'ssim_brain_n2n', 'psnr_brain_n2n', 'mae_brain_ddpm', 'rmse_brain_ddpm', 'ssim_brain_ddpm', 'psnr_brain_ddpm', 'mae_brain_ddpm_avg', 'rmse_brain_ddpm_avg', 'ssim_brain_ddpm_avg', 'psnr_brain_ddpm_avg', 'mae_motion', 'rmse_motion', 'ssim_motion', 'psnr_motion', 'mae_n2n', 'rmse_n2n', 'ssim_n2n', 'psnr_n2n', 'mae_ddpm', 'rmse_ddpm', 'ssim_ddpm', 'psnr_ddpm', 'mae_ddpm_avg', 'rmse_ddpm_avg', 'ssim_ddpm_avg', 'psnr_ddpm_avg'])
    df.to_excel('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/quantitative_results.xlsx', index = False)



