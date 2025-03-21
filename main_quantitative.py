import sys
sys.path.append('/workspace/Documents')
import os
import torch
import lpips
import numpy as np 
import nibabel as nb
import pandas as pd
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Data_processing as Data_processing

build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 3)

def compute_lpips_3d(prediction, ground_truth, max_val = None, min_val = None, net_type='vgg'):
    assert prediction.shape == ground_truth.shape, "Shape mismatch between prediction and ground truth!"
    
    # Convert to float32
    prediction = prediction.astype(np.float32)
    ground_truth = ground_truth.astype(np.float32)

    # Normalize to [-1, 1] range as required by LPIPS
    if max_val == None:
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 2 - 1 
    else:
        prediction = (prediction - min_val) / (max_val - min_val) * 2 - 1
    if max_val == None:
        ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min()) * 2 - 1
    else:
        ground_truth = (ground_truth - min_val) / (max_val - min_val) * 2 - 1

    # Initialize LPIPS loss model
    loss_fn = lpips.LPIPS(net=net_type).to('cuda' if torch.cuda.is_available() else 'cpu')

    lpips_scores = []
    
    # Loop through each slice along the z-axis
    for i in range(prediction.shape[2]):
        pred_slice = prediction[:, :, i]  # Get 2D slice
        gt_slice = ground_truth[:, :, i]  # Get corresponding GT slice

        # Convert numpy arrays to torch tensors
        pred_tensor = torch.tensor(pred_slice).unsqueeze(0).unsqueeze(0)  # Shape: [1,1,H,W]
        gt_tensor = torch.tensor(gt_slice).unsqueeze(0).unsqueeze(0)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred_tensor = pred_tensor.to(device)
        gt_tensor = gt_tensor.to(device)
        loss_fn = loss_fn.to(device)

        # Compute LPIPS score for this slice
        lpips_score = loss_fn(pred_tensor, gt_tensor)
        lpips_scores.append(lpips_score.item())

    # Compute average LPIPS score across all slices
    return np.mean(lpips_scores)
        
avg_slice = True
print('avg_slice:', avg_slice)
results = []
for i in range(0,n.shape[0]):
    patient_id = patient_id_list[n[i]]
    patient_subid = patient_subid_list[n[i]]
    random_n = random_num_list[n[i]]
    print(patient_id, patient_subid, random_n)

    gt_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch70_1/gt_img.nii.gz')
    gt_img = nb.load(gt_file).get_fdata()
    # process gt
    shape = gt_img.shape
    gt_img_new = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2]-2))
    for i in range(1, gt_img.shape[2]-1):
        gt_img_new[:,:,i-1] = np.mean(gt_img[:,:,i-1:i+2], axis = 2)
    gt_img = np.copy(gt_img_new) if avg_slice else np.copy(gt_img)
    gt_img_brain = Data_processing.cutoff_intensity(gt_img, cutoff_low=-100, cutoff_high=100)

    condition_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch70_1/condition_img.nii.gz')
    condition_img = nb.load(condition_file).get_fdata() if avg_slice == False else nb.load(condition_file).get_fdata()[:,:,1:shape[2]-1]
    condition_img_brain = Data_processing.cutoff_intensity(condition_img, cutoff_low=-100, cutoff_high=100)

    noise2noise_file = os.path.join('/mnt/camca_NAS/denoising/models/noise2noise_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch77/pred_img.nii.gz')
    noise2noise_img = nb.load(noise2noise_file).get_fdata() if avg_slice == False else nb.load(noise2noise_file).get_fdata()[:,:,1:shape[2]-1]
    noise2noise_img_brain = Data_processing.cutoff_intensity(noise2noise_img, cutoff_low=-100, cutoff_high=100)

    # noise2noise_avg_file = os.path.join('/mnt/camca_NAS/denoising/models/noise2noise_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'final_avg/pred_img.nii.gz')
    # noise2noise_avg_img = nb.load(noise2noise_avg_file).get_fdata()
    # noise2noise_avg_img_brain = Data_processing.cutoff_intensity(noise2noise_avg_img, cutoff_low=-100, cutoff_high=100)

    supervised_file = os.path.join('/mnt/camca_NAS/denoising/models/supervised_DDPM_possion_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch50_1/pred_img.nii.gz')
    supervised_img = nb.load(supervised_file).get_fdata() if avg_slice == False else nb.load(supervised_file).get_fdata()[:,:,1:shape[2]-1]
    supervised_img_brain = Data_processing.cutoff_intensity(supervised_img, cutoff_low=-100, cutoff_high=100)

    supervised_avg_file = os.path.join('/mnt/camca_NAS/denoising/models/supervised_DDPM_possion_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch50final/pred_img.nii.gz')
    supervised_avg_img = nb.load(supervised_avg_file).get_fdata() if avg_slice == False else nb.load(supervised_avg_file).get_fdata()[:,:,1:shape[2]-1]
    supervised_avg_img_brain = Data_processing.cutoff_intensity(supervised_avg_img, cutoff_low=-100, cutoff_high=100)

    ddpm_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch70_1/pred_img.nii.gz')
    ddpm_img = nb.load(ddpm_file).get_fdata() if avg_slice == False else nb.load(ddpm_file).get_fdata()[:,:,1:shape[2]-1]
    ddpm_img_brain = Data_processing.cutoff_intensity(ddpm_img, cutoff_low=-100, cutoff_high=100)

    ddpm_avg_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch70final/pred_img.nii.gz')
    ddpm_avg_img = nb.load(ddpm_avg_file).get_fdata() if avg_slice == False else nb.load(ddpm_avg_file).get_fdata()[:,:,1:shape[2]-1]
    ddpm_avg_img_brain = Data_processing.cutoff_intensity(ddpm_avg_img, cutoff_low=-100, cutoff_high=100)

    # compare brain region
    mae_brain_motion, _, rmse_brain_motion, _, ssim_brain_motion,psnr_brain_motion = ff.compare(condition_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
    mae_brain_n2n, _, rmse_brain_n2n, _, ssim_brain_n2n,psnr_brain_n2n = ff.compare(noise2noise_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
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

    # print('all image:')
    # print('motion:', mae_motion, rmse_motion, ssim_motion, psnr_motion)
    # print('n2n:', mae_n2n, rmse_n2n, ssim_n2n, psnr_n2n)
    # print('supervised:', mae_supervised, rmse_supervised, ssim_supervised, psnr_supervised)
    # print('supervised_avg:', mae_supervised_avg, rmse_supervised_avg, ssim_supervised_avg, psnr_supervised_avg)
    # print('ddpm:', mae_ddpm, rmse_ddpm, ssim_ddpm, psnr_ddpm)
    # print('ddpm_avg:', mae_ddpm_avg, rmse_ddpm_avg, ssim_ddpm_avg, psnr_ddpm_avg)

    # calculate lpips in brain
    lpips_brain_motion = compute_lpips_3d(condition_img_brain, gt_img_brain, max_val = 100, min_val = 0)
    lpips_brain_n2n = compute_lpips_3d(noise2noise_img_brain, gt_img_brain, max_val = 100, min_val = 0)
    lpips_brain_supervised = compute_lpips_3d(supervised_img_brain, gt_img_brain, max_val = 100, min_val = 0)
    lpips_brain_supervised_avg = compute_lpips_3d(supervised_avg_img_brain, gt_img_brain, max_val = 100, min_val = 0)
    lpips_brain_ddpm = compute_lpips_3d(ddpm_img_brain, gt_img_brain, max_val = 100, min_val = 0)
    lpips_brain_ddpm_avg = compute_lpips_3d(ddpm_avg_img_brain, gt_img_brain, max_val = 100, min_val = 0)

    print('lpips: ')
    print('motion:', lpips_brain_motion)
    print('n2n:', lpips_brain_n2n)
    print('supervised:', lpips_brain_supervised)
    print('supervised_avg:', lpips_brain_supervised_avg)
    print('ddpm:', lpips_brain_ddpm)
    print('ddpm_avg:', lpips_brain_ddpm_avg)

    results.append([patient_id, patient_subid, random_n, mae_brain_motion, rmse_brain_motion, ssim_brain_motion, psnr_brain_motion, mae_brain_n2n, rmse_brain_n2n, ssim_brain_n2n, psnr_brain_n2n, mae_brain_supervised, rmse_brain_supervised, ssim_brain_supervised, psnr_brain_supervised, mae_brain_supervised_avg, rmse_brain_supervised_avg, ssim_brain_supervised_avg, psnr_brain_supervised_avg, mae_brain_ddpm, rmse_brain_ddpm, ssim_brain_ddpm, psnr_brain_ddpm, mae_brain_ddpm_avg, rmse_brain_ddpm_avg, ssim_brain_ddpm_avg, psnr_brain_ddpm_avg, mae_motion, rmse_motion, ssim_motion, psnr_motion, mae_n2n, rmse_n2n, ssim_n2n, psnr_n2n, mae_supervised, rmse_supervised, ssim_supervised, psnr_supervised, mae_supervised_avg, rmse_supervised_avg, ssim_supervised_avg, psnr_supervised_avg, mae_ddpm, rmse_ddpm, ssim_ddpm, psnr_ddpm, mae_ddpm_avg, rmse_ddpm_avg, ssim_ddpm_avg, psnr_ddpm_avg, lpips_brain_motion, lpips_brain_n2n, lpips_brain_supervised, lpips_brain_supervised_avg, lpips_brain_ddpm, lpips_brain_ddpm_avg])
    df = pd.DataFrame(results, columns = ['patient_id', 'patient_subid', 'random_n', 'mae_brain_motion', 'rmse_brain_motion', 'ssim_brain_motion', 'psnr_brain_motion', 'mae_brain_n2n', 'rmse_brain_n2n', 'ssim_brain_n2n', 'psnr_brain_n2n', 'mae_brain_supervised', 'rmse_brain_supervised', 'ssim_brain_supervised', 'psnr_brain_supervised', 'mae_brain_supervised_avg', 'rmse_brain_supervised_avg', 'ssim_brain_supervised_avg', 'psnr_brain_supervised_avg', 'mae_brain_ddpm', 'rmse_brain_ddpm', 'ssim_brain_ddpm', 'psnr_brain_ddpm', 'mae_brain_ddpm_avg', 'rmse_brain_ddpm_avg', 'ssim_brain_ddpm_avg', 'psnr_brain_ddpm_avg', 'mae_motion', 'rmse_motion', 'ssim_motion', 'psnr_motion', 'mae_n2n', 'rmse_n2n', 'ssim_n2n', 'psnr_n2n', 'mae_supervised', 'rmse_supervised', 'ssim_supervised', 'psnr_supervised', 'mae_supervised_avg', 'rmse_supervised_avg', 'ssim_supervised_avg', 'psnr_supervised_avg', 'mae_ddpm', 'rmse_ddpm', 'ssim_ddpm', 'psnr_ddpm', 'mae_ddpm_avg', 'rmse_ddpm_avg', 'ssim_ddpm_avg', 'psnr_ddpm_avg', 'lpips_brain_motion', 'lpips_brain_n2n', 'lpips_brain_supervised', 'lpips_brain_supervised_avg', 'lpips_brain_ddpm', 'lpips_brain_ddpm_avg'])
    file_name = 'quantitative_results.xlsx' if avg_slice == False else 'quantitative_results_avg_slice.xlsx'
    df.to_excel(os.path.join('/mnt/camca_NAS/denoising/models', file_name), index = False)

    

