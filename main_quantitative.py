import sys
sys.path.append('/workspace/Documents')
import os
import torch
import lpips
import numpy as np 
import nibabel as nb
import pandas as pd
from scipy.ndimage import binary_erosion, generate_binary_structure
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Data_processing as Data_processing

build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian_NAS.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 2)

def compute_lpips_3d(prediction, ground_truth,mask = None, max_val = None, min_val = None, net_type='vgg'):
    assert prediction.shape == ground_truth.shape, "Shape mismatch between prediction and ground truth!"
    
    # Convert to float32
    prediction = prediction.astype(np.float32)
    ground_truth = ground_truth.astype(np.float32)

    if mask is not None:
        prediction[mask==0] = np.min(prediction)
        ground_truth[mask==0] = np.min(ground_truth)

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

### metric calculations
# results = []; results_mean = []
# for i in range(0,n.shape[0]):
#     patient_id = patient_id_list[n[i]]
#     patient_subid = patient_subid_list[n[i]]
#     random_n = random_num_list[n[i]]
#     print(patient_id, patient_subid, random_n)

#     # reference image
#     gt_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta0/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch61_1/gt_img.nii.gz')
#     gt_img = nb.load(gt_file).get_fdata()
#     gt_img_brain = Data_processing.cutoff_intensity(gt_img, cutoff_low=-100, cutoff_high=100)

#     # noisy image
#     condition_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta0/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch61_1/condition_img.nii.gz')
#     condition_img = nb.load(condition_file).get_fdata()
#     condition_img_brain = Data_processing.cutoff_intensity(condition_img, cutoff_low=-100, cutoff_high=100)

#     # noise2noise
#     noise2noise_file = os.path.join('/mnt/camca_NAS/denoising/models/noise2noise/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch78/pred_img.nii.gz')
#     noise2noise_img = nb.load(noise2noise_file).get_fdata() 
#     noise2noise_img_brain = Data_processing.cutoff_intensity(noise2noise_img, cutoff_low=-100, cutoff_high=100)

#     # supervised method
#     supervised_file = os.path.join('/mnt/camca_NAS/denoising/models/supervised_possion/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch58_1/pred_img.nii.gz')
#     supervised_img = nb.load(supervised_file).get_fdata() 
#     supervised_img_brain = Data_processing.cutoff_intensity(supervised_img, cutoff_low=-100, cutoff_high=100)

#     # supervised_avg_file = os.path.join('/mnt/camca_NAS/denoising/models/supervised_DDPM_possion_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch50final/pred_img.nii.gz')
#     # supervised_avg_img = nb.load(supervised_avg_file).get_fdata() 
#     # supervised_avg_img_brain = Data_processing.cutoff_intensity(supervised_avg_img, cutoff_low=-100, cutoff_high=100)

#     # our method (unsupervised), beta = 0, 1 inference
#     unsupervised_beta0_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta0/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch61_1/pred_img.nii.gz')
#     unsupervised_beta0_img = nb.load(unsupervised_beta0_file).get_fdata()
#     unsupervised_beta0_img_brain = Data_processing.cutoff_intensity(unsupervised_beta0_img, cutoff_low=-100, cutoff_high=100)

#     # our method (unsupervised), beta = 0, 10 inference
#     unsupervised_beta0_avg10_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta0/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch61avg/pred_img_scans10.nii.gz')
#     unsupervised_beta0_avg10_img = nb.load(unsupervised_beta0_avg10_file).get_fdata()
#     unsupervised_beta0_avg10_img_brain = Data_processing.cutoff_intensity(unsupervised_beta0_avg10_img, cutoff_low=-100, cutoff_high=100)

#     # our method (unsupervised), beta = 0, 20 inference
#     unsupervised_beta0_avg20_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta0/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch61avg/pred_img_scans20.nii.gz')
#     unsupervised_beta0_avg20_img = nb.load(unsupervised_beta0_avg20_file).get_fdata()
#     unsupervised_beta0_avg20_img_brain = Data_processing.cutoff_intensity(unsupervised_beta0_avg20_img, cutoff_low=-100, cutoff_high=100)

#     # our method (unsupervised), beta = 10, 1 inference
#     unsupervised_beta10_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta10/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch65_1/pred_img.nii.gz')
#     unsupervised_beta10_img = nb.load(unsupervised_beta10_file).get_fdata()
#     unsupervised_beta10_img_brain = Data_processing.cutoff_intensity(unsupervised_beta10_img, cutoff_low=-100, cutoff_high=100)

#     # our method (unsupervised), beta = 10, 10 inference
#     unsupervised_beta10_avg10_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta10/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch65avg/pred_img_scans10.nii.gz')
#     unsupervised_beta10_avg10_img = nb.load(unsupervised_beta10_avg10_file).get_fdata()
#     unsupervised_beta10_avg10_img_brain = Data_processing.cutoff_intensity(unsupervised_beta10_avg10_img, cutoff_low=-100, cutoff_high=100)

#     # our method (unsupervised), beta = 10, 20 inference
#     unsupervised_beta10_avg20_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta10/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch65avg/pred_img_scans20.nii.gz')
#     unsupervised_beta10_avg20_img = nb.load(unsupervised_beta10_avg20_file).get_fdata()
#     unsupervised_beta10_avg20_img_brain = Data_processing.cutoff_intensity(unsupervised_beta10_avg20_img, cutoff_low=-100, cutoff_high=100)

#     # our method (unsupervised), beta = 20, 1 inference
#     unsupervised_beta20_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta20/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch58_1/pred_img.nii.gz')
#     unsupervised_beta20_img = nb.load(unsupervised_beta20_file).get_fdata()
#     unsupervised_beta20_img_brain = Data_processing.cutoff_intensity(unsupervised_beta20_img, cutoff_low=-100, cutoff_high=100)

#     #### TASK 1: compare the mean value in brain region (crop a ROI in the center)
#     x,y = 256,256
#     mean_gt = np.mean(np.clip(gt_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100))
#     mean_condition = np.mean(np.clip(condition_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100))
#     mean_noise2noise = np.mean(np.clip(noise2noise_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100))
#     mean_supervised= np.mean(np.clip(supervised_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100))
#     mean_unsupervised_beta0 = np.mean(np.clip(unsupervised_beta0_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100))
#     mean_unsupervised_beta0_avg10 = np.mean(np.clip(unsupervised_beta0_avg10_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100))
#     mean_unsupervised_beta0_avg20 = np.mean(np.clip(unsupervised_beta0_avg20_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100))
#     mean_unsupervised_beta10 = np.mean(np.clip(unsupervised_beta10_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100))
#     mean_unsupervised_beta10_avg10 = np.mean(np.clip(unsupervised_beta10_avg10_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100))
#     mean_unsupervised_beta10_avg20 = np.mean(np.clip(unsupervised_beta10_avg20_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100))
#     mean_unsupervised_beta20 = np.mean(np.clip(unsupervised_beta20_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100))

#     results_mean.append([patient_id, patient_subid, random_n,
#     mean_gt, mean_condition, mean_noise2noise, mean_supervised,
#     mean_unsupervised_beta0, mean_unsupervised_beta0_avg10, mean_unsupervised_beta0_avg20,
#     mean_unsupervised_beta10, mean_unsupervised_beta10_avg10, mean_unsupervised_beta10_avg20,
#     mean_unsupervised_beta20])
#     df_mean = pd.DataFrame(results_mean, columns = ['patient_id', 'patient_subid', 'random_n',
#     'mean_gt', 'mean_condition', 'mean_noise2noise', 'mean_supervised',
#     'mean_unsupervised_beta0', 'mean_unsupervised_beta0_avg10', 'mean_unsupervised_beta0_avg20',
#     'mean_unsupervised_beta10', 'mean_unsupervised_beta10_avg10', 'mean_unsupervised_beta10_avg20',
#     'mean_unsupervised_beta20'])
#     file_name = 'mean_measurements.xlsx'
#     df_mean.to_excel(os.path.join('/mnt/camca_NAS/denoising/models', file_name), index = False)

    
#     #### TASK2: compare brain region metrics 
#     # define eroded mask
#     mask = np.zeros(gt_img_brain.shape, dtype=bool)
#     mask[(gt_img_brain>0) & (gt_img_brain < 100)] = 1
#     structure = np.ones((6,6))
#     mask_eroded = np.zeros_like(mask, dtype=bool)
#     for i in range(mask.shape[2]):
#         mask_eroded[:, :, i] = binary_erosion(mask[:, :, i], structure=structure, iterations=1)

#     mae_motion, _, rmse_motion, _, ssim_motion,_ = ff.compare(condition_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
#     mae_n2n, _, rmse_n2n, _, ssim_n2n, _ = ff.compare(noise2noise_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
#     mae_supervised, _, rmse_supervised, _, ssim_supervised,_= ff.compare(supervised_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)

#     mae_unsupervised_beta0, _, rmse_unsupervised_beta0, _, ssim_unsupervised_beta0,_ = ff.compare(unsupervised_beta0_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
#     mae_unsupervised_beta0_avg10, _, rmse_unsupervised_beta0_avg10, _, ssim_unsupervised_beta0_avg10,_ = ff.compare(unsupervised_beta0_avg10_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
#     mae_unsupervised_beta0_avg20, _, rmse_unsupervised_beta0_avg20, _, ssim_unsupervised_beta0_avg20,_ = ff.compare(unsupervised_beta0_avg20_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
#     mae_unsupervised_beta10, _, rmse_unsupervised_beta10, _, ssim_unsupervised_beta10,_ = ff.compare(unsupervised_beta10_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
#     mae_unsupervised_beta10_avg10, _, rmse_unsupervised_beta10_avg10, _, ssim_unsupervised_beta10_avg10,_ = ff.compare(unsupervised_beta10_avg10_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
#     mae_unsupervised_beta10_avg20, _, rmse_unsupervised_beta10_avg20, _, ssim_unsupervised_beta10_avg20,_ = ff.compare(unsupervised_beta10_avg20_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
#     mae_unsupervised_beta20, _, rmse_unsupervised_beta20, _, ssim_unsupervised_beta20,_ = ff.compare(unsupervised_beta20_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)


#     print('motion metrics: ', mae_motion, rmse_motion, ssim_motion)
#     print('n2n metrics: ', mae_n2n, rmse_n2n, ssim_n2n)
#     print('supervised metrics: ', mae_supervised, rmse_supervised, ssim_supervised)
#     print('unsupervised beta0 metrics: ', mae_unsupervised_beta0, rmse_unsupervised_beta0, ssim_unsupervised_beta0)
#     print('unsupervised beta0 avg10 metrics: ', mae_unsupervised_beta0_avg10, rmse_unsupervised_beta0_avg10, ssim_unsupervised_beta0_avg10)
#     print('unsupervised beta0 avg20 metrics: ', mae_unsupervised_beta0_avg20, rmse_unsupervised_beta0_avg20, ssim_unsupervised_beta0_avg20)
#     print('unsupervised beta10 metrics: ', mae_unsupervised_beta10, rmse_unsupervised_beta10, ssim_unsupervised_beta10)
#     print('unsupervised beta10 avg10 metrics: ', mae_unsupervised_beta10_avg10, rmse_unsupervised_beta10_avg10, ssim_unsupervised_beta10_avg10)
#     print('unsupervised beta10 avg20 metrics: ', mae_unsupervised_beta10_avg20, rmse_unsupervised_beta10_avg20, ssim_unsupervised_beta10_avg20)
#     print('unsupervised beta20 metrics: ', mae_unsupervised_beta20, rmse_unsupervised_beta20, ssim_unsupervised_beta20)

#     ##### TASK2b: calculate lpips in brain
#     lpips_motion = compute_lpips_3d(condition_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
#     lpips_n2n = compute_lpips_3d(noise2noise_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
#     lpips_supervised = compute_lpips_3d(supervised_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)

#     lpips_unsupervised_beta0 = compute_lpips_3d(unsupervised_beta0_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
#     lpips_unsupervised_beta0_avg10 = compute_lpips_3d(unsupervised_beta0_avg10_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
#     lpips_unsupervised_beta0_avg20 = compute_lpips_3d(unsupervised_beta0_avg20_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
#     lpips_unsupervised_beta10 = compute_lpips_3d(unsupervised_beta10_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
#     lpips_unsupervised_beta10_avg10 = compute_lpips_3d(unsupervised_beta10_avg10_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
#     lpips_unsupervised_beta10_avg20 = compute_lpips_3d(unsupervised_beta10_avg20_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
#     lpips_unsupervised_beta20 = compute_lpips_3d(unsupervised_beta20_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)

#     print('lpips motion:', lpips_motion)
#     print('lpips n2n:', lpips_n2n)
#     print('lpips supervised:', lpips_supervised)
#     print('lpips unsupervised beta0:', lpips_unsupervised_beta0)
#     print('lpips unsupervised beta0 avg10:', lpips_unsupervised_beta0_avg10)
#     print('lpips unsupervised beta0 avg20:', lpips_unsupervised_beta0_avg20)
#     print('lpips unsupervised beta10:', lpips_unsupervised_beta10)
#     print('lpips unsupervised beta10 avg10:', lpips_unsupervised_beta10_avg10)
#     print('lpips unsupervised beta10 avg20:', lpips_unsupervised_beta10_avg20)
#     print('lpips unsupervised beta20:', lpips_unsupervised_beta20)

#     results.append([patient_id, patient_subid, random_n, 
#     mae_motion, rmse_motion, ssim_motion, lpips_motion,
#     mae_n2n, rmse_n2n, ssim_n2n, lpips_n2n,
#     mae_supervised, rmse_supervised, ssim_supervised, lpips_supervised,
#     mae_unsupervised_beta0, rmse_unsupervised_beta0, ssim_unsupervised_beta0, lpips_unsupervised_beta0,
#     mae_unsupervised_beta0_avg10, rmse_unsupervised_beta0_avg10, ssim_unsupervised_beta0_avg10, lpips_unsupervised_beta0_avg10,
#     mae_unsupervised_beta0_avg20, rmse_unsupervised_beta0_avg20, ssim_unsupervised_beta0_avg20, lpips_unsupervised_beta0_avg20,
#     mae_unsupervised_beta10, rmse_unsupervised_beta10, ssim_unsupervised_beta10, lpips_unsupervised_beta10,
#     mae_unsupervised_beta10_avg10, rmse_unsupervised_beta10_avg10, ssim_unsupervised_beta10_avg10, lpips_unsupervised_beta10_avg10,
#     mae_unsupervised_beta10_avg20, rmse_unsupervised_beta10_avg20, ssim_unsupervised_beta10_avg20, lpips_unsupervised_beta10_avg20,
#     mae_unsupervised_beta20, rmse_unsupervised_beta20, ssim_unsupervised_beta20, lpips_unsupervised_beta20])
    

#     df = pd.DataFrame(results, columns = ['patient_id', 'patient_subid', 'random_n', 
#     'mae_motion', 'rmse_motion', 'ssim_motion', 'lpips_motion',
#     'mae_n2n', 'rmse_n2n', 'ssim_n2n', 'lpips_n2n',
#     'mae_supervised', 'rmse_supervised', 'ssim_supervised', 'lpips_supervised',
#     'mae_unsupervised_beta0', 'rmse_unsupervised_beta0', 'ssim_unsupervised_beta0', 'lpips_unsupervised_beta0',
#     'mae_unsupervised_beta0_avg10', 'rmse_unsupervised_beta0_avg10', 'ssim_unsupervised_beta0_avg10', 'lpips_unsupervised_beta0_avg10',
#     'mae_unsupervised_beta0_avg20', 'rmse_unsupervised_beta0_avg20', 'ssim_unsupervised_beta0_avg20', 'lpips_unsupervised_beta0_avg20',
#     'mae_unsupervised_beta10', 'rmse_unsupervised_beta10', 'ssim_unsupervised_beta10', 'lpips_unsupervised_beta10',
#     'mae_unsupervised_beta10_avg10', 'rmse_unsupervised_beta10_avg10', 'ssim_unsupervised_beta10_avg10', 'lpips_unsupervised_beta10_avg10',
#     'mae_unsupervised_beta10_avg20', 'rmse_unsupervised_beta10_avg20', 'ssim_unsupervised_beta10_avg20', 'lpips_unsupervised_beta10_avg20',
#     'mae_unsupervised_beta20', 'rmse_unsupervised_beta20', 'ssim_unsupervised_beta20', 'lpips_unsupervised_beta20'])
#     file_name = 'quantitative_results.xlsx' 
#     df.to_excel(os.path.join('/mnt/camca_NAS/denoising/models', file_name), index = False)

### effect of the number of inferences
results_inference = [] 
for i in range(0,n.shape[0]):
    patient_id = patient_id_list[n[i]]
    patient_subid = patient_subid_list[n[i]]
    random_n = random_num_list[n[i]]
    print(patient_id, patient_subid, random_n)
    r = [patient_id, patient_subid, random_n]

    gt_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta0/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch61_1/gt_img.nii.gz')
    gt_img = nb.load(gt_file).get_fdata()
    gt_img_brain = Data_processing.cutoff_intensity(gt_img, cutoff_low=-100, cutoff_high=100)

    files = ff.sort_timeframe(ff.find_all_target_files(['pred_img_scans*'], os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta0/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch61avg')),2,'s','.')
    
    for k in range(0,len(files)):
        img = nb.load(files[k]).get_fdata() 
        img_brain = Data_processing.cutoff_intensity(img, cutoff_low=-100, cutoff_high=100)
        mae, _, rmsm, _, ssim,_ = ff.compare(img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
        lpips = compute_lpips_3d(img_brain, gt_img_brain, max_val = 100, min_val = 0)
        print('when avg num:', k+1, ' mae:', mae, 'rmse:', rmsm, 'ssim:', ssim, 'lpips:', lpips)
        r += [mae, rmsm, ssim, lpips]

    results_inference.append(r)

    columns = ['patient_id', 'patient_subid', 'random_n']
    for k in range(0,len(files)):
        columns += ['mae_inference'+str(k+1), 'rmse_inference'+str(k+1), 'ssim_inference'+str(k+1), 'lpips_inference'+str(k+1)]
    df = pd.DataFrame(results_inference, columns = columns)
    file_name = 'quantitative_results_multiple_inferences.xlsx' 
    df.to_excel(os.path.join('/mnt/camca_NAS/denoising/models', file_name), index = False)

    

   