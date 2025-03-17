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
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 3)

# def optimize(pred,gt):
#     best_mae = 100; best_x = 0; best_y = 0; best_pred = np.copy(pred)
#     for x in range(-3,4):
#         for y in range(-3,4):
#             # translate pred in [x,y]
#             pred_new = np.copy(pred)
#             pred_new = np.roll(pred_new, x, axis = 0)
#             pred_new = np.roll(pred_new, y, axis = 1)
#             mae,_,_,_,_,_ = ff.compare(pred_new,gt, cutoff_low = 0, cutoff_high = 100)
#             if mae < best_mae:
#                 best_mae = mae
#                 best_x = x
#                 best_y = y
#                 best_pred = np.copy(pred_new)
#             print(x,y,mae)
#     return best_x, best_y, best_pred
        

results = []
for i in range(0,5):#n.shape[0]):
    patient_id = patient_id_list[n[i]]
    patient_subid = patient_subid_list[n[i]]
    random_n = random_num_list[n[i]]
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

    # noise2noise_avg_file = os.path.join('/mnt/camca_NAS/denoising/models/noise2noise_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'final_avg/pred_img.nii.gz')
    # noise2noise_avg_img = nb.load(noise2noise_avg_file).get_fdata()
    # noise2noise_avg_img_brain = Data_processing.cutoff_intensity(noise2noise_avg_img, cutoff_low=-100, cutoff_high=100)

    supervised_file = os.path.join('/mnt/camca_NAS/denoising/models/supervised_DDPM_possion_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch50_1/pred_img.nii.gz')
    supervised_img = nb.load(supervised_file).get_fdata()
    supervised_img_brain = Data_processing.cutoff_intensity(supervised_img, cutoff_low=-100, cutoff_high=100)

    supervised_avg_file = os.path.join('/mnt/camca_NAS/denoising/models/supervised_DDPM_possion_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch50final/pred_img.nii.gz')
    supervised_avg_img = nb.load(supervised_avg_file).get_fdata()
    supervised_avg_img_brain = Data_processing.cutoff_intensity(supervised_avg_img, cutoff_low=-100, cutoff_high=100)

    ddpm_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch70_1/pred_img.nii.gz')
    ddpm_img = nb.load(ddpm_file).get_fdata()
    ddpm_img_brain = Data_processing.cutoff_intensity(ddpm_img, cutoff_low=-100, cutoff_high=100)

    ddpm_avg_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch70final/pred_img.nii.gz')
    ddpm_avg_img = nb.load(ddpm_avg_file).get_fdata()
    ddpm_avg_img_brain = Data_processing.cutoff_intensity(ddpm_avg_img, cutoff_low=-100, cutoff_high=100)

    # compare brain region
    mae_brain_motion, _, rmse_brain_motion, _, ssim_brain_motion,psnr_brain_motion = ff.compare(condition_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
    mae_brain_n2n, _, rmse_brain_n2n, _, ssim_brain_n2n,psnr_brain_n2n = ff.compare(noise2noise_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
    # mae_brain_n2n_avg, _, rmse_brain_n2n_avg, _, ssim_brain_n2n_avg,psnr_brain_n2n_avg = ff.compare(noise2noise_avg_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
    mae_brain_supervised, _, rmse_brain_supervised, _, ssim_brain_supervised,psnr_brain_supervised = ff.compare(supervised_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
    mae_brain_supervised_avg, _, rmse_brain_supervised_avg, _, ssim_brain_supervised_avg,psnr_brain_supervised_avg = ff.compare(supervised_avg_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)

    mae_brain_ddpm, _, rmse_brain_ddpm, _, ssim_brain_ddpm,psnr_brain_ddpm = ff.compare(ddpm_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
    mae_brain_ddpm_avg, _, rmse_brain_ddpm_avg, _, ssim_brain_ddpm_avg,psnr_brain_ddpm_avg = ff.compare(ddpm_avg_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)

    print('motion:', mae_brain_motion, rmse_brain_motion, ssim_brain_motion, psnr_brain_motion)
    print('n2n:', mae_brain_n2n, rmse_brain_n2n, ssim_brain_n2n, psnr_brain_n2n)
    print('mae_brain_supervised:', mae_brain_supervised, rmse_brain_supervised, ssim_brain_supervised, psnr_brain_supervised)
    print('mae_brain_supervised_avg:', mae_brain_supervised_avg, rmse_brain_supervised_avg, ssim_brain_supervised_avg, psnr_brain_supervised_avg)
    print('ddpm:', mae_brain_ddpm, rmse_brain_ddpm, ssim_brain_ddpm, psnr_brain_ddpm)
    print('ddpm_avg:', mae_brain_ddpm_avg, rmse_brain_ddpm_avg, ssim_brain_ddpm_avg, psnr_brain_ddpm_avg)

    # compare all 
    mae_motion, _, rmse_motion, _, ssim_motion,psnr_motion = ff.compare(condition_img, gt_img, cutoff_low = -100)
    mae_n2n, _, rmse_n2n, _, ssim_n2n,psnr_n2n = ff.compare(noise2noise_img, gt_img, cutoff_low = -100)
    mae_supervised, _, rmse_supervised, _, ssim_supervised,psnr_supervised = ff.compare(supervised_img, gt_img, cutoff_low = -100)
    mae_supervised_avg, _, rmse_supervised_avg, _, ssim_supervised_avg,psnr_supervised_avg = ff.compare(supervised_avg_img, gt_img, cutoff_low = -100)
    mae_ddpm, _, rmse_ddpm, _, ssim_ddpm,psnr_ddpm = ff.compare(ddpm_img, gt_img, cutoff_low = -100)
    mae_ddpm_avg, _, rmse_ddpm_avg, _, ssim_ddpm_avg,psnr_ddpm_avg = ff.compare(ddpm_avg_img, gt_img, cutoff_low = -100)

    print('all image:')
    print('motion:', mae_motion, rmse_motion, ssim_motion, psnr_motion)
    print('n2n:', mae_n2n, rmse_n2n, ssim_n2n, psnr_n2n)
    print('supervised:', mae_supervised, rmse_supervised, ssim_supervised, psnr_supervised)
    print('supervised_avg:', mae_supervised_avg, rmse_supervised_avg, ssim_supervised_avg, psnr_supervised_avg)
    print('ddpm:', mae_ddpm, rmse_ddpm, ssim_ddpm, psnr_ddpm)
    print('ddpm_avg:', mae_ddpm_avg, rmse_ddpm_avg, ssim_ddpm_avg, psnr_ddpm_avg)

    results.append([patient_id, patient_subid, random_n, mae_brain_motion, rmse_brain_motion, ssim_brain_motion, psnr_brain_motion, mae_brain_n2n, rmse_brain_n2n, ssim_brain_n2n, psnr_brain_n2n, mae_brain_supervised, rmse_brain_supervised, ssim_brain_supervised, psnr_brain_supervised, mae_brain_supervised_avg, rmse_brain_supervised_avg, ssim_brain_supervised_avg, psnr_brain_supervised_avg, mae_brain_ddpm, rmse_brain_ddpm, ssim_brain_ddpm, psnr_brain_ddpm, mae_brain_ddpm_avg, rmse_brain_ddpm_avg, ssim_brain_ddpm_avg, psnr_brain_ddpm_avg, mae_motion, rmse_motion, ssim_motion, psnr_motion, mae_n2n, rmse_n2n, ssim_n2n, psnr_n2n, mae_supervised, rmse_supervised, ssim_supervised, psnr_supervised, mae_supervised_avg, rmse_supervised_avg, ssim_supervised_avg, psnr_supervised_avg, mae_ddpm, rmse_ddpm, ssim_ddpm, psnr_ddpm, mae_ddpm_avg, rmse_ddpm_avg, ssim_ddpm_avg, psnr_ddpm_avg])
    df = pd.DataFrame(results, columns = ['patient_id', 'patient_subid', 'random_n', 'mae_brain_motion', 'rmse_brain_motion', 'ssim_brain_motion', 'psnr_brain_motion', 'mae_brain_n2n', 'rmse_brain_n2n', 'ssim_brain_n2n', 'psnr_brain_n2n', 'mae_brain_supervised', 'rmse_brain_supervised', 'ssim_brain_supervised', 'psnr_brain_supervised', 'mae_brain_supervised_avg', 'rmse_brain_supervised_avg', 'ssim_brain_supervised_avg', 'psnr_brain_supervised_avg', 'mae_brain_ddpm', 'rmse_brain_ddpm', 'ssim_brain_ddpm', 'psnr_brain_ddpm', 'mae_brain_ddpm_avg', 'rmse_brain_ddpm_avg', 'ssim_brain_ddpm_avg', 'psnr_brain_ddpm_avg', 'mae_motion', 'rmse_motion', 'ssim_motion', 'psnr_motion', 'mae_n2n', 'rmse_n2n', 'ssim_n2n', 'psnr_n2n', 'mae_supervised', 'rmse_supervised', 'ssim_supervised', 'psnr_supervised', 'mae_supervised_avg', 'rmse_supervised_avg', 'ssim_supervised_avg', 'psnr_supervised_avg', 'mae_ddpm', 'rmse_ddpm', 'ssim_ddpm', 'psnr_ddpm', 'mae_ddpm_avg', 'rmse_ddpm_avg', 'ssim_ddpm_avg', 'psnr_ddpm_avg'])
    df.to_excel('/mnt/camca_NAS/denoising/models/quantitative_results.xlsx')

    

