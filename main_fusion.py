# we use diffusion to generate 20 samples, the next question is how to fuse them
# we have avg, weighted sum and fft

import sys
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM as edm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator as Generator
from scipy.ndimage import gaussian_filter
from skimage.util import view_as_windows

build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [0]) 
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 3)
print('total number:', n.shape[0])

fusion_method = 'fft'
if fusion_method == 'fft':
    radius = 100
    alpha = 0.5

if fusion_method == 'weighted':
    h_param = 0.05

for i in range(0, 1):#n.shape[0]):

    patient_id = patient_id_list[n[i]]
    patient_subid = patient_subid_list[n[i]]
    print(f"Processing patient {patient_id} {patient_subid}...")

    folder = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images',patient_id,patient_subid, 'random_0')
    folders = ff.find_all_target_files(['epoch70_*'],folder)
    # print('folders:', folders)

    if fusion_method == 'avg':
        save_folder = os.path.join('/mnt/camca_NAS/denoising/models', 'unsupervised_DDPM_gaussian_2D', 'pred_images', patient_id, patient_subid, 'random_0/epoch70avg')
    elif fusion_method == 'fft':
        save_folder = os.path.join('/mnt/camca_NAS/denoising/models', 'unsupervised_DDPM_gaussian_2D', 'pred_images', patient_id, patient_subid, 'random_0/epoch70fft_r' + str(radius))
    elif fusion_method == 'weighted':
        save_folder = os.path.join('/mnt/camca_NAS/denoising/models', 'unsupervised_DDPM_gaussian_2D', 'pred_images', patient_id, patient_subid, 'random_0/epoch70weighted_h' + str(h_param))

    ff.make_folder([save_folder])

    total_n = len(folders)
    loaded_data = np.zeros([ total_n, 512, 512, 50])

    for j in range(0,total_n):
        loaded_data[j] = nb.load(os.path.join(folders[j],'pred_img.nii.gz')).get_fdata()
        if j == 0:
            affine = nb.load(os.path.join(folders[j],'pred_img.nii.gz')).affine

    for scan_num in [2,5,10,20]:#range(1,total_predicts+1):
        print('scan_num:', scan_num)
        loaded_image = np.zeros((scan_num, 512,512,50))

        for j in range(scan_num):
            loaded_image[j] = loaded_data[j]

        if fusion_method == 'avg':
            final_img = np.mean(loaded_image, axis = 3)
        
        if fusion_method == 'weighted':
            final_image_patch_weighted = np.zeros([512,512,50])
            for i in range(0,10):
                slice_range = [i * 5, (i + 1) * 5]
                image= np.transpose(np.copy(loaded_image[:,:,:,slice_range[0]:slice_range[1]])/1000,(1,2,3,0))

                h,w,s,N = image.shape
                patch_size = (5,5,3)
                rH, rW, rS = patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2

                # Step 1: Pad each sample image to handle border cases
                padded_stack = np.pad(image, pad_width=((rH, rH), (rW, rW), (rS, rS), (0, 0)), mode='constant',constant_values=np.min(image))

                # Step 2: Extract patches for each pixel from each sample
                # For each of the 20 images, get view_as_windows: shape becomes [h, w, N, patch_size, patch_size]
                patches = np.stack([view_as_windows(padded_stack[:, :, :, k], patch_size) for k in range(N)], axis=3)  # stack along sample axis

                # Step 3: Compute the mean patch across samples, shape: [h, w, patch_size, patch_size]
                mean_patch = np.mean(patches, axis=3)

                # Step 4: Compute distance from each sample patch to the mean patch
                # Euclidean distance for each sample patch to the mean
                dists = np.sum((patches - mean_patch[:, :, :, None, :, :, :]) ** 2, axis=(-1, -2,-3))  # shape: [h, w, N]

                # Step 5: Convert distances to weights using Gaussian kernel
                weights = np.exp(-dists / (h_param ** 2))  # shape: [h, w, N]
                weights_sum = np.sum(weights, axis=3, keepdims=True)
                weights_sum[weights_sum == 0] = 1e-6  # avoid divide by zero
                weights /= weights_sum  # normalize weights, shape: [H, W, S, N]

                # Step 6: Weighted fusion: for each pixel location, do weighted sum over 20 samples
                fused_image_weighted = np.sum(image * weights, axis=3)  

                final_image_patch_weighted[:,:,slice_range[0]:slice_range[1]] = fused_image_weighted * 1000
            final_img = final_image_patch_weighted

        if fusion_method == 'fft':
            final_image_fft = np.zeros([512,512,50])
            final_low_freq, final_high_freq = np.zeros([512,512,50]), np.zeros([512,512,50])

            for slice_index in range(0,final_image_fft.shape[2]):
                image = np.copy(loaded_image[:,:,:,slice_index])/1000
                image = np.transpose(image, (1,2,0))

                # Perform FFT
                fft_stack = np.fft.fft2(image, axes=(0, 1))
                fft_stack_shifted = np.fft.fftshift(fft_stack, axes=(0, 1))

                lowpass_mask = ff.create_2d_lowpass_mask((image.shape[0], image.shape[1]), radius=radius)
                lowpass_mask = lowpass_mask[:,:,None]

                # Apply lowpass filter
                low_freq = fft_stack_shifted * lowpass_mask
                high_freq = fft_stack_shifted * (1 - lowpass_mask)

                low_freq = np.fft.ifftshift(low_freq, axes=(0,1))
                high_freq = np.fft.ifftshift(high_freq, axes=(0,1))
                # print('low_freq:', low_freq.shape, ' high_freq:', high_freq.shape)

                low_fused = np.mean(low_freq, axis=2)    
                high_fused = np.mean(high_freq, axis=2)

                fft_fused = low_fused + alpha * high_fused
                fused_image = np.fft.ifftn(fft_fused, axes=(0, 1)).real * 1000
                final_image_fft[:,:,slice_index] = fused_image

                final_low_freq[:,:,slice_index] = np.fft.ifftn(low_fused, axes=(0, 1)).real * 1000
                final_high_freq[:,:,slice_index] = np.fft.ifftn(high_fused, axes=(0, 1)).real * 1000
            final_img = final_image_fft

        # save
        save_file = os.path.join(save_folder, 'pred_img_scans' + str(scan_num) + '.nii.gz')
        nb.save(nb.Nifti1Image(final_img, affine), save_file)
        if fusion_method == 'fft':
            nb.save(nb.Nifti1Image(final_low_freq, affine), os.path.join(save_folder, 'low_freq_scans' + str(scan_num) + '.nii.gz'))
            nb.save(nb.Nifti1Image(final_high_freq, affine), os.path.join(save_folder, 'high_freq_scans' + str(scan_num) + '.nii.gz'))
        
