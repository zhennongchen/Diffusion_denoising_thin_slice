#!/usr/bin/env python

# %%
import numpy as np
import nibabel as nb 
import pandas as pd
import os
import copy
import Diffusion_denoising_thin_slice.simulation.ct_basic_PCD as ct
import Diffusion_denoising_thin_slice.functions_collection as ff
import ct_projector.projector.numpy.parallel as ct_para

main_path = '/mnt/camca_NAS/denoising/Data'

def interleave_filter_and_recon(projector, prjs, custom_filter,angles):
    # interleave the filter
    projector = copy.deepcopy(projector)
    prjs = prjs.copy()

    # make sure the detector center is within the central pixel (usually 0.25 of a pixel)
    # so that when interleaving the detectors the length will be twice the original
    offu = int(projector.off_u)
    projector.off_u -= offu
    projector.nu -= 2 * offu
    prjs = prjs[..., :-2 * offu]

    # interleave the projections
    new_prjs = np.zeros(list(prjs.shape[:-1]) + [prjs.shape[-1] * 2])
    for iview in range(new_prjs.shape[1]):
        iview_opp = (iview + new_prjs.shape[1] // 2) % new_prjs.shape[1]
        new_prjs[:, iview, :, 1::2] = prjs[:, iview, :, :]
        new_prjs[:, iview, :, 0::2] = prjs[:, iview_opp, :, ::-1]
    prjs = new_prjs
    projector.off_u = 0
    projector.nu = prjs.shape[-1]
    projector.du = projector.du / 2

    # build rl filter
    nu = prjs.shape[-1]
    du = projector.du
    rl_filter = np.zeros([2 * nu - 1], np.float32)
    k = np.arange(len(rl_filter)) - (nu - 1)
    for i in range(len(rl_filter)):
        if k[i] == 0:
            rl_filter[i] = 1 / (4 * du * du)
        elif k[i] % 2 != 0:
            rl_filter[i] = -1 / (np.pi * np.pi * k[i] * k[i] * du * du)
    frl_filter = np.fft.fft(rl_filter, len(custom_filter))
    frl_filter = np.abs(frl_filter)

    frl_filter = frl_filter * len(frl_filter) / prjs.shape[1] * du * 2

    custom_filter = frl_filter * custom_filter

    # filter the projection
    fprj = np.fft.fft(prjs, len(custom_filter), axis=-1)
    fprj = fprj * custom_filter
    fprj = np.fft.ifft(fprj, axis=-1)[..., :prjs.shape[-1]]
    fprj = fprj.real.astype(np.float32) * np.pi / len(custom_filter) / 2
    fprj = np.copy(fprj, 'C')

    # reconstruction
    recon = ct_para.pixel_driven_bp(projector, fprj, angles)

    return recon

# set a noise range
noise_sheet = pd.read_excel(os.path.join(os.path.dirname(main_path),'Patient_lists','portable_CT_CNR.xlsx'),dtype={'Patient_ID': str, 'Patient_subID': str})
CNR_array = noise_sheet['CNR'].values
std_array = noise_sheet['background_std'].values
print('CNR max, min, mean, std, median, lower quartile, upper quartile: ', np.max(CNR_array), np.min(CNR_array), np.mean(CNR_array), np.std(CNR_array), np.median(CNR_array), np.percentile(CNR_array, 25), np.percentile(CNR_array, 75))
print('std max, min, mean, std, median, lower quartile, upper quartile: ', np.max(std_array), np.min(std_array), np.mean(std_array), np.std(std_array), np.median(std_array), np.percentile(std_array, 25), np.percentile(std_array, 75))

# CNR range: 5-8.6, mean+std = 6.6+-0.7, [6.1,6.9]
# background noise range: 4-22.8, mean+std = 10.5+-3, [8.8, 11.9]

# define the patient list
patient_sheet = pd.read_excel(os.path.join('/mnt/camca_NAS/denoising/','Patient_lists', 'fixedCT_static.xlsx'),dtype={'Patient_ID': str, 'Patient_subID': str})
print('patient sheet len: ', len(patient_sheet))

for i in range(0,1):# len(patient_sheet)):
    row = patient_sheet.iloc[i]
    patient_id = row['Patient_ID']
    patient_subID = row['Patient_subID']

    print(patient_id, patient_subID)

    save_folder_case = os.path.join(main_path,'simulation_v2', patient_id,patient_subID)
    ff.make_folder([os.path.dirname(save_folder_case), save_folder_case])

    # for dose factor in range 0.30-0.9 with step 0.05
    # possion_hann_dose_range = [0.60,0.80]
    # gaussian_custom_dose_range = [0.085,0.1]

    for noise_type in ['gaussian']:
    
        for k in range(0,1):
            # dose_factor = round(dose_factor,3)
            # if noise_type == 'possion':
            #     dose_factor = np.random.uniform(possion_hann_dose_range[0],possion_hann_dose_range[1] + 1e-8)
            # elif noise_type == 'gaussian':
            #     dose_factor = np.random.uniform(gaussian_custom_dose_range[0],gaussian_custom_dose_range[1] + 1e-8)
            # print('dose factor: ', dose_factor)
            # important
            # cp.cuda.Device(0).use()
            # ct_projector.set_device(0)

            # img file
            img_file = os.path.join(main_path,'fixedCT',patient_id,patient_subID,'img_thinslice.nii.gz')
            print(img_file)
            # load img
            img_clean = nb.load(img_file).get_fdata().astype(np.float32)
            img_clean[img_clean < -1024] = -1024
            spacing = nb.load(img_file).header.get_zooms()[::-1]
            affine = nb.load(img_file).affine

            # process img
            img0 = img_clean.copy()
            img0 = img0[:,:,50:51]
            img0 = np.rollaxis(img0,-1,0)
            print('img shape, min, max: ', img0.shape, np.min(img0), np.max(img0))
            print('spacing: ', spacing)
      
            # define projectors
            projector = ct.define_forward_projector_pcd(img0,spacing, file_name = './pcd_parallel_6x5_512.cfg')

            # FP
            # set angles
            angles = projector.get_angles()
           
            for slice_n in range(0, 1):
                img_slice = img0[[slice_n],:,:].copy()
                img_slice = (img_slice[np.newaxis, ...] + 1000) / 1000 * 0.019 

                prjs = ct_para.distance_driven_fp(projector, img_slice, angles)
                fprjs = ct_para.ramp_filter(projector, prjs, 'rl')
            #     recon_slice = ct_para.distance_driven_bp(projector, fprjs, angles, True)
            #     recon_slice = recon_slice[0,0] / 0.019 * 1000 - 1000
            #     recon[:,:,slice_n] = recon_slice

            # print('recon shape, min, max: ', recon.shape, np.min(recon), np.max(recon))

            # nb.save(nb.Nifti1Image(recon, affine), os.path.join(save_folder_case,'recon_fp.nii.gz'))
            nb.save(nb.Nifti1Image(img_clean[:,:,50:51], affine), os.path.join(save_folder_case,'img.nii.gz'))

            # add noise
            if noise_type[0:2] == 'po':
                # Poisson noise
                noise_of_prjs = ct.add_poisson_noise(prjs, N0=1000000, dose_factor = 0.2) - prjs
            elif noise_type[0:2] == 'ga':
                noise_of_prjs = ct.add_gaussian_noise(prjs, N0=1000000, dose_factor=0.2) - prjs

            # recon
    
            if noise_type[0:2] == 'po':
                img_slice = img0[[0],:,:].copy()
                img_slice = (img_slice[np.newaxis, ...] + 1000) / 1000 * 0.019 
                # hann filter
                fnoise = ct_para.ramp_filter(projector, noise_of_prjs, 'hann')
                recon_hann = ct_para.distance_driven_bp(projector, fnoise, angles, True) + img_slice
                recon_hann = recon_hann[0, 0] / 0.019 * 1000 - 1000
                recon_hann = recon_hann[:,:,np.newaxis]
        
                print('recon hann shape, min, max: ', recon_hann.shape, np.min(recon_hann), np.max(recon_hann))
                nb.save(nb.Nifti1Image(recon_hann, affine), os.path.join(save_folder_case,'recon_hann.nii.gz'))
                
            elif noise_type[0:2] == 'ga':
                img_slice = img0[[0],:,:].copy()
                img_slice = (img_slice[np.newaxis, ...] + 1000) / 1000 * 0.019 
                # custom filter
                custom_additional_filter = ct.get_additional_filter_to_rl(os.path.join('/mnt/camca_NAS/denoising/Data', 'softTissueKernel_65'), projector.nu, projector.du, projector.nview)
                recon_custom = interleave_filter_and_recon(projector, noise_of_prjs, custom_additional_filter,angles) + img_slice
                recon_custom = recon_custom[0, 0] / 0.019 * 1000 - 1000
                recon_custom = recon_custom[:,:,np.newaxis]
                print('recon custom shape, min, max: ', recon_custom.shape, np.min(recon_custom), np.max(recon_custom))
                nb.save(nb.Nifti1Image(recon_custom, affine), os.path.join(save_folder_case,'recon_custom.nii.gz'))
                
            #     custom_filter = np.fromfile(os.path.join('/mnt/camca_NAS/denoising/Data', 'softTissueKernel_65'), np.float32)
            #     filter_len = 2048
            #     new_filter = custom_filter[:len(custom_filter) // 2]
    
            #     custom_filter = np.zeros([filter_len], np.float32)
            #     custom_filter[:len(new_filter)] = new_filter
            #     custom_filter[len(custom_filter) // 2:] = custom_filter[len(custom_filter) // 2:0:-1]
    
            #     proj_noise_copy = np.copy(proj_noise, 'C')
   
            #     fprjs = np.fft.fft(proj_noise_copy, len(custom_filter), axis=-1)
            #     fprjs = fprjs * custom_filter
            #     fprjs = np.fft.ifft(fprjs, axis=-1)[..., :proj_noise_copy.shape[-1]]
            #     fprjs = fprjs.real.astype(np.float32) * np.pi / len(custom_filter) / 2
            #     fprj_noise = np.copy(fprjs, 'C')

            # # backprojection
            # projector.set_backprojector(ct_fan.distance_driven_bp, angles=cuangles, is_fbp=True) 
            # cufprj_noise= cp.array(fprj_noise, cp.float32, order = 'C')
            # curecon_noise = projector.bp(cufprj_noise)
            # recon_noise = curecon_noise.get()
            # recon_noise = recon_noise[:,0,...]

            # # add original img
            # # recon_noise = recon_noise /0.019 * 1000 - 1024 
            # # recon_original = img[0,...] /0.019 * 1000 - 1024
            # recon = recon_noise + img[0,...]
            # recon = recon /0.019 * 1000 - 1024
            # recon_noise = recon_noise /0.019 * 1000 


            # save_folder = os.path.join(save_folder_case, noise_type +'_random_' + str(k))
            # ff.make_folder([save_folder])

            # # save recon
            # recon_nb_image = np.rollaxis(recon,0,3)
            # # recon_nb_image = recon_nb_image[:,:,recon_nb_image.shape[-1]//2 - 25: recon_nb_image.shape[-1]//2 + 25]
            # nb.save(nb.Nifti1Image(recon_nb_image,img_affine), os.path.join(save_folder,'recon.nii.gz'))
            # # recon_noise_image = np.rollaxis(recon_noise,0,3)
            # nb.save(nb.Nifti1Image(recon_noise_image,img_affine), os.path.join(save_folder,'recon_noise.nii.gz'))
            # recon_original_image = recon_nb_image - recon_noise_image
            # nb.save(nb.Nifti1Image(recon_original_image,img_affine), os.path.join(save_folder,'recon_original.nii.gz'))
        

    # # CNR range: 5-8.6, mean+std = 6.6+-0.7, [6.1,6.9]
    # # background noise range: 4-22.8, mean+std = 10.5+-3, [8.8, 11.9]

    # # check CNR
    # # put threshold [-100,100]
    # img_cutoff = Data_processing.cutoff_intensity(recon_nb_image, cutoff_low=-100, cutoff_high=100)
    # print('min:', np.min(img_cutoff), 'max:', np.max(img_cutoff))
    # size = img_cutoff.shape

    # # calculate CNR
    # a = img_cutoff[size[0]//2-50:size[0]//2+50,size[1]//2-50:size[1]//2+50,size[2]//2 -50 :   size[2]//2+50]
    # CNR = 100/np.std(a[ (a> 0) & (a < 100)])
    # print('CNR: ', CNR)

    # # check background noise
    # region = recon_nb_image[size[0]-30: size[0], size[1] - 30: size[1], size[2]//2-10:size[2]//2+10]
    # background_std = np.std(region[region < 0])
    # print('background noise: ', background_std)




