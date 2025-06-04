#!/usr/bin/env python

# this is the script for new data collected at 2024/04
# %%
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import glob as gb
import nibabel as nb 
import math
import pandas as pd
import os
import copy
from skimage.measure import block_reduce
import Diffusion_denoising_thin_slice.simulation.ct_basic_PCD as ct

import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Data_processing as Data_processing

import ct_projector.projector.cupy as ct_projector
import CTProjector.src.ct_projector.projector.cupy as ct_projector
import CTProjector.src.ct_projector.projector.cupy.fan_equiangular as ct_fan 
import CTProjector.src.ct_projector.projector.numpy as numpy_projector
import CTProjector.src.ct_projector.projector.numpy.fan_equiangluar as numpy_fan

main_path = '/mnt/camca_NAS/denoising/Data'

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

    save_folder_case = os.path.join(main_path,'simulation', patient_id,patient_subID)
    ff.make_folder([os.path.dirname(save_folder_case), save_folder_case])

    # for dose factor in range 0.30-0.9 with step 0.05
    possion_hann_dose_range = [0.60,0.80]
    gaussian_custom_dose_range = [0.085,0.1]

    for noise_type in ['possion','gaussian']:
    
        for k in range(4,5):
            # dose_factor = round(dose_factor,3)
            if noise_type == 'possion':
                dose_factor = np.random.uniform(possion_hann_dose_range[0],possion_hann_dose_range[1] + 1e-8)
            elif noise_type == 'gaussian':
                dose_factor = np.random.uniform(gaussian_custom_dose_range[0],gaussian_custom_dose_range[1] + 1e-8)
            print('dose factor: ', dose_factor)
            # important
            cp.cuda.Device(0).use()
            ct_projector.set_device(0)

            # img file
            img_file = os.path.join(main_path,'fixedCT',patient_id,patient_subID,'img_thinslice.nii.gz')

            # load img
            img,spacing,img_affine = ct.basic_image_processing(img_file)
            print('img shape, min, max: ', img.shape, np.min(img), np.max(img))
            print('spacing: ', spacing)

            # define projectors
            img = img[np.newaxis, ...]
            print('image shape: ', img.shape)
            projector = ct.define_forward_projector(img,spacing,total_view = 1440)
            fbp_projector = ct.backprojector(img,spacing)

            # FP
            # set angles
            angles = ct.get_angles_zc(1440, 360 ,0)
            proj = ct.fp_static(img,angles,projector, geometry = 'fan')

            # add noise
            if noise_type[0:2] == 'po':
                # Poisson noise
                proj_noise = ct.add_poisson_noise(proj, N0=1000000, dose_factor=dose_factor) - proj
            elif noise_type[0:2] == 'ga':
                proj_noise = ct.add_gaussian_noise(proj, N0=1000000, dose_factor=dose_factor) - proj

            # recon
            cuangles = cp.array(angles, cp.float32, order = 'C')

            # filter
            if noise_type[0:2] == 'po':
                # hann filter
                fprj_noise = numpy_fan.ramp_filter(fbp_projector, proj_noise, filter_type='hann')
            elif noise_type[0:2] == 'ga':
                # custom filter
                
                custom_filter = np.fromfile(os.path.join('/mnt/camca_NAS/denoising/Data', 'softTissueKernel_65'), np.float32)
                filter_len = 2048
                new_filter = custom_filter[:len(custom_filter) // 2]
    
                custom_filter = np.zeros([filter_len], np.float32)
                custom_filter[:len(new_filter)] = new_filter
                custom_filter[len(custom_filter) // 2:] = custom_filter[len(custom_filter) // 2:0:-1]
    
                proj_noise_copy = np.copy(proj_noise, 'C')
   
                fprjs = np.fft.fft(proj_noise_copy, len(custom_filter), axis=-1)
                fprjs = fprjs * custom_filter
                fprjs = np.fft.ifft(fprjs, axis=-1)[..., :proj_noise_copy.shape[-1]]
                fprjs = fprjs.real.astype(np.float32) * np.pi / len(custom_filter) / 2
                fprj_noise = np.copy(fprjs, 'C')

            # backprojection
            projector.set_backprojector(ct_fan.distance_driven_bp, angles=cuangles, is_fbp=True) 
            cufprj_noise= cp.array(fprj_noise, cp.float32, order = 'C')
            curecon_noise = projector.bp(cufprj_noise)
            recon_noise = curecon_noise.get()
            recon_noise = recon_noise[:,0,...]

            # add original img
            # recon_noise = recon_noise /0.019 * 1000 - 1024 
            # recon_original = img[0,...] /0.019 * 1000 - 1024
            recon = recon_noise + img[0,...]
            recon = recon /0.019 * 1000 - 1024
            recon_noise = recon_noise /0.019 * 1000 


            save_folder = os.path.join(save_folder_case, noise_type +'_random_' + str(k))
            ff.make_folder([save_folder])

            # save recon
            recon_nb_image = np.rollaxis(recon,0,3)
            # recon_nb_image = recon_nb_image[:,:,recon_nb_image.shape[-1]//2 - 25: recon_nb_image.shape[-1]//2 + 25]
            nb.save(nb.Nifti1Image(recon_nb_image,img_affine), os.path.join(save_folder,'recon.nii.gz'))
            # recon_noise_image = np.rollaxis(recon_noise,0,3)
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









# main_folder = '/mnt/camca_NAS/Portable_CT_data'

# motion_type = 'simulated_partial_motion_v2' # each gantry rotation has motion except the first one
# amplitude_max_severe = 10 #8
# displacement_max_severe = 6
# amplitude_max_mild = 5 #5
# displacement_max_mild = 3

# motion_freq = 0.5 # 0.3 or 0.35
# severe_freq = 0.5 # 0.6 or 0.65
# double_skull_freq = 0.25 # 0.3 or 0.35

# change_direction_limit = 2
# CP_num = 5

# geometry = 'fan'
# total_view = 1400  ### 2340 views by default
# gantry_rotation_time = 500 #unit ms, 500ms by default
# view_increment = 28 # increment in gantry views

# # define the patient list
# patient_sheet = pd.read_excel(os.path.join(main_folder,'Patient_list', 'NEW_CT_concise_collected_fixed_static_edited.xlsx'),dtype={'Patient_ID': str, 'Patient_subID': str})
# patient_sheet['use'] = patient_sheet['use'].fillna(0)
# patient_sheet = patient_sheet[(patient_sheet['use'] != 0) & (patient_sheet['use'] != 'no')]
# print('patient sheet len: ', len(patient_sheet))

# data_folder = os.path.join(main_folder, 'nii_imgs_202404', 'static')
# save_folder = os.path.join(main_folder, 'simulations_202404', motion_type)
# ff.make_folder([save_folder])

# # define patient list index and simulation index
# L = np.arange(6,16)
# patient_index_list = np.arange(0,20)

# for i in patient_index_list:
#     row = patient_sheet.iloc[i]
#     patient_id = row['Patient_ID']
#     patient_subid = row['Patient_subID']
#     print('\n',i, patient_id, patient_subid)

#     save_folder_patient = os.path.join(save_folder, patient_id, patient_subid)
#     ff.make_folder([os.path.join(save_folder, patient_id), save_folder_patient])

#     img_file = ff.find_all_target_files(['fixed/img_1mm.nii.gz'],os.path.join(data_folder, patient_id, patient_subid))
#     if len(img_file) != 1:
#         ValueError('no raw data')
    
#     img,spacing,img_affine = ct.basic_image_processing(img_file[0])
#     print('nib image shape: ',img.shape, ' spacing: ',spacing)

#     # define projectors
#     img = img[np.newaxis, ...]
#     projector = ct.define_forward_projector(img,spacing,total_view)
#     fbp_projector = ct.backprojector(img,spacing)
    
#     # very important - make sure that the arrays are saved in C order
#     cp.cuda.Device(0).use()
#     ct_projector.set_device(0)

#     # load the static image as reference
#     if os.path.isfile(os.path.join(main_folder, 'simulations_202404','simulated_all_motion_v1',patient_id, patient_subid, 'static','image_data','recon.nii.gz')) == 0:
#         static_ref = None
#     else:
#         static_ref= nb.load(os.path.join(main_folder, 'simulations_202404','simulated_all_motion_v1',patient_id, patient_subid, 'static','image_data','recon.nii.gz')).get_fdata()
#         static_ref = np.rollaxis(static_ref,2,0)
#         print('static ref shape: ', static_ref.shape)
#     print('static_ref is None? ', static_ref is None)

#     for random_i in L:
#         t = np.linspace(0, gantry_rotation_time, CP_num, endpoint=True)
#         # create folder
#         if random_i == 0:
#             random_folder = os.path.join(save_folder_patient,'static')
#         else:
#             random_folder = os.path.join(save_folder_patient,'random_' +str(random_i))
#         ff.make_folder([random_folder, os.path.join(random_folder,'image_data')])
#         print('\n',random_i  , 'random')

#         if os.path.isfile(os.path.join(random_folder,'image_data','recon_resample.nii.gz')) == 1:
#             print('already done this motion')

#             motion = nb.load(os.path.join(random_folder, 'image_data','recon_resample.nii.gz')).get_fdata()
#             static = nb.load(os.path.join(main_folder, 'simulations_202404','simulated_all_motion_v1',patient_id, patient_subid, 'static','image_data','recon_resample.nii.gz')).get_fdata()
#             mae, mse, rmse, r_rmse, ssim = ff.compare(motion[:,:,5:], static[:,:,5:], cutoff_low = -100)
#             print('mae: ', mae,  ' rmse: ', rmse, ' ssim: ', ssim)
#             continue

#         # gantry coverage = 1cm, which means it rotates once for every 1cm interval in the z-axis
#         # set a sga reference
#         sga_list = []; sga_reference = int(np.random.uniform(0,90))
#         # second, find out how many rotations we need
#         print('spacing: ', spacing, ' img shape: ', img.shape)
#         rotation_num = int(spacing[0] * img.shape[1] // 10)
#         slice_coverage = 10
#         if random_i == 0:
#             rotation_num = 1
#             slice_coverage = int(spacing[0] * img.shape[1] // 10) * 10
#         print('rotation_num: ', rotation_num, ' slice_coverage: ', slice_coverage)

#         # start to generate the motion for each rotation
#         # set whether static or motion for each rotation
#         if random_i != 0:
#             while True:
#                 motion_status = [np.random.uniform(0,1) <motion_freq for k in range(rotation_num-2)]
#                 if motion_freq <=0:
#                     break
#                 if np.sum(motion_status) > 0:
#                     break
#             motion_status = [False] + motion_status + [False] # add two static rotations at the beginning and end
#         if random_i == 0: # static
#             motion_status = [False] 
#         print('motion_status: ', motion_status)
        
#         amplitude_collect = np.zeros([rotation_num, CP_num, 6])
#         recon = np.zeros([slice_coverage * rotation_num,img.shape[2],img.shape[3]])
#         projection = np.zeros([slice_coverage * rotation_num, total_view, 1 , projector.nu])

#         for rot_n in range(0,rotation_num):
#             # first set the SGA:
#             sga = sga_reference + int(np.random.uniform(-5,5))
#             sga = max(0, sga); sga = min(90, sga)
#             sga_list.append(sga)
#             print('rotation: ', rot_n, 'sga: ', sga)
#             # set the partial image:
#             slice_start = rot_n * slice_coverage
#             slice_end = (rot_n + 1) * slice_coverage
#             slice_prior = max(0, slice_start - slice_coverage * 2)
#             slice_post = min(img.shape[1], slice_end + slice_coverage * 2)
#             print('slice start: ', slice_start, ' slice end: ', slice_end, ' slice prior: ', slice_prior, ' slice post: ', slice_post)
#             img_partial = img[:,slice_prior:slice_post,:]

#             # set the motion 
#             # first: determine whether in this rotation we have motion or static, and whether it's severe or mild
#             if motion_status[rot_n]:
#                 amplitude_max_tem_r = amplitude_max_mild # rotation always max =  5 degree
#                 displacement_max_tem_r = displacement_max_mild

#                 if np.random.uniform(0,1) < severe_freq: # severe motion
#                     severe_motion = True
#                     amplitude_max_tem_t = amplitude_max_severe
#                     displacement_max_tem_t = displacement_max_severe
#                 else: # mild motion
#                     severe_motion = False
#                     amplitude_max_tem_t = amplitude_max_mild
#                     displacement_max_tem_t = displacement_max_mild
#             else:
#                 severe_motion = False
#                 amplitude_max_tem_t, displacement_max_tem_t, amplitude_max_tem_r, displacement_max_tem_r = 0, 0, 0, 0
#             print('severe motion: ', severe_motion)
#             # second: determine whether the initial pose has been changed compared to last rotation
#             offset_values = [0] * 6
            
#             # third: generate the motion
#             # translation
#             amplitude_txty_mm = transform.motion_control_point_generation(2, CP_num, amplitude_max = amplitude_max_tem_t, displacement_max = displacement_max_tem_t, change_direction_limit = change_direction_limit, offset_value = offset_values[0:2], print_result =False)
#             amplitude_tx_mm = amplitude_txty_mm[:,0]
#             amplitude_ty_mm = amplitude_txty_mm[:,1]
#             if severe_motion:
#                 amplitude_tz_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 3, displacement_max = 1.5, change_direction_limit = change_direction_limit, offset_value = offset_values[2], print_result =False)[:,0]
#             elif not severe_motion and motion_status[rot_n]:
#                 amplitude_tz_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 2, displacement_max = 1, change_direction_limit = change_direction_limit, offset_value = offset_values[2], print_result =False)[:,0]
#             else:
#                 amplitude_tz_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 0, displacement_max = 0, change_direction_limit = change_direction_limit, offset_value = offset_values[2], print_result =False)[:,0]
            
#             # rotations
#             while True:
#                 amplitude_rxryrz_degree = transform.motion_control_point_generation(3, CP_num, amplitude_max = amplitude_max_tem_r, displacement_max = displacement_max_tem_r, change_direction_limit = change_direction_limit, offset_value = offset_values[3:6], print_result =False)
#                 amplitude_rx_degree = amplitude_rxryrz_degree[:,0]
#                 amplitude_ry_degree = amplitude_rxryrz_degree[:,1]
#                 amplitude_rz_degree = amplitude_rxryrz_degree[:,2]
#                 if np.max(abs(amplitude_rx_degree - amplitude_rx_degree[0]))+ np.max(abs(amplitude_ry_degree - amplitude_ry_degree[0])) <= 6:
#                     break

#             # let's also consider the double skull artifacts, espeically in the occipital bone
#             if np.random.uniform(0,1) < double_skull_freq and motion_status[rot_n]:
#                 print('yes we have double skull')
#                 while True:
#                     amplitude_ty_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 12, displacement_max = 8, change_direction_limit = change_direction_limit, offset_value = offset_values[1], print_result =False)[:,0]
#                     amplitude_tx_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 5, displacement_max = 3, change_direction_limit = change_direction_limit, offset_value = offset_values[0], print_result =False)[:,0]
#                     amplitude_tz_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 1, displacement_max = 0.5, change_direction_limit = change_direction_limit, offset_value = offset_values[2], print_result =False)[:,0]

#                     amplitude_rxryrz_degree = transform.motion_control_point_generation(3, CP_num, amplitude_max = 3, displacement_max = 1, change_direction_limit = change_direction_limit, offset_value = offset_values[3:6], print_result =False)
#                     amplitude_rx_degree = amplitude_rxryrz_degree[:,0]
#                     amplitude_ry_degree = amplitude_rxryrz_degree[:,1]
#                     amplitude_rz_degree = amplitude_rxryrz_degree[:,2]
#                     if np.max(abs(amplitude_ty_mm - amplitude_ty_mm[0])) >= 7:
#                         break

#             print('amplitude_tx_mm: ', amplitude_tx_mm)
#             print('amplitude_ty_mm: ', amplitude_ty_mm)
#             print('amplitude_tz_mm: ', amplitude_tz_mm)
#             print('amplitude_rx_degree: ', amplitude_rx_degree)
#             print('amplitude_ry_degree: ', amplitude_ry_degree)
#             print('amplitude_rz_degree: ', amplitude_rz_degree)
     
#             # save the motion parameters
#             parameter_file = os.path.join(random_folder,'motion_parameters.txt')
#             if rot_n == 0:
#                 ff.txt_writer(parameter_file, True, [t.tolist(),amplitude_tx_mm, amplitude_ty_mm, amplitude_tz_mm, amplitude_rx_degree, amplitude_ry_degree, amplitude_rz_degree, [sga],[total_view],[gantry_rotation_time]],['time_points','translation_x_CP','translation_y_CP','translation_z_CP', 'rotation_x_CP', 'rotation_y_CP','rotation_z_CP','starting_gantry_angle', 'total_projection_view','gantry_rotation_time(ms)'])
#             else:
#                 ff.txt_writer(parameter_file, False, [t.tolist(),amplitude_tx_mm, amplitude_ty_mm, amplitude_tz_mm, amplitude_rx_degree, amplitude_ry_degree, amplitude_rz_degree, [sga],[total_view],[gantry_rotation_time]],['time_points','translation_x_CP','translation_y_CP','translation_z_CP', 'rotation_x_CP', 'rotation_y_CP','rotation_z_CP','starting_gantry_angle', 'total_projection_view','gantry_rotation_time(ms)'])
            
#             collect = np.stack([amplitude_tx_mm, amplitude_ty_mm, amplitude_tz_mm, amplitude_rx_degree, amplitude_ry_degree, amplitude_rz_degree], axis = 1)
#             amplitude_collect[rot_n,...] = collect
            
#             # prepare spline fit
#             spline_tx = transform.interp_func(t, np.asarray([i/spacing[1] for i in amplitude_tx_mm]))
#             spline_ty = transform.interp_func(t, np.asarray([i/spacing[2] for i in amplitude_ty_mm]))
#             spline_tz = transform.interp_func(t, np.asarray([i/spacing[0] for i in amplitude_tz_mm]))
#             spline_rx = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_rx_degree]))
#             spline_ry = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_ry_degree]))
#             spline_rz = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_rz_degree]))

#             angles = ff.get_angles_zc(total_view, 360 ,sga)

#             # if static and has the static image ref, then no need to do the projection
#             if motion_status[rot_n] == False:
#                 if static_ref is not None:
#                     recon[slice_coverage * (rot_n) : slice_coverage * (rot_n + 1),...] = static_ref[slice_coverage*rot_n:slice_coverage*(rot_n+1),...]
#                     continue

#             # generate forward projection
#             projection_partial = ct.fp_w_spline_motion_model(img_partial, projector, angles, spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz, geometry, total_view = total_view, gantry_rotation_time = gantry_rotation_time, slice_num = None, increment = view_increment, order = 3)
#             projection_partial = projection_partial[slice_start - slice_prior : slice_end - slice_prior,...]
            
#             # generate backprojection
#             recon_partial = ct.filtered_backporjection(projection_partial,angles,projector,fbp_projector, geometry, back_to_original_value=True)
#             recon[slice_coverage * (rot_n) : slice_coverage * (rot_n + 1),...] = recon_partial
            
#             # save fp
#             # projection_save_version = nb.Nifti1Image(projection[:,:,0,:], img_affine)
#             # nb.save(projection_save_version, os.path.join(random_folder,'projection.nii.gz'))

#         parameter_file = os.path.join(random_folder,'motion_parameters.npy')
#         np.save(parameter_file, np.array([[amplitude_collect],[sga_list], [t], [total_view], [gantry_rotation_time]], dtype=object))

#         # generate backprojection
#         # recon = ct.filtered_backporjection(projection,angles,projector,fbp_projector, geometry, back_to_original_value=True)
           
#         # save recon
#         recon_nb_image = np.rollaxis(recon,0,3)  
#         print('under 1mm, recon shape: ', recon_nb_image.shape)
#         print('max and min value: ', np.max(recon_nb_image), np.min(recon_nb_image))
#         nb.save(nb.Nifti1Image(recon_nb_image,img_affine), os.path.join(random_folder,'image_data','recon.nii.gz'))

#         # resample recon
#         recon_1mm = nb.load(os.path.join(random_folder,'image_data','recon.nii.gz'))
#         new_dim = [1,1,2.5]
#         recon_resample = ff.resample_nifti(recon_1mm, order=3,  mode = 'nearest',  cval = np.min(recon_1mm.get_fdata()), in_plane_resolution_mm=new_dim[0], slice_thickness_mm=new_dim[-1])
#         recon_resample = nb.Nifti1Image(recon_resample.get_fdata(), affine=recon_resample.affine, header=recon_resample.header)
#         print('after 1mm, recon shape: ', recon_resample.get_fdata().shape)
        
#         nb.save(recon_resample, os.path.join(random_folder,'image_data','recon_resample.nii.gz'))

#         # get some statics
#         motion = nb.load(os.path.join(random_folder, 'image_data','recon_resample.nii.gz')).get_fdata()
#         static = nb.load(os.path.join(main_folder, 'simulations_202404','simulated_all_motion_v1',patient_id, patient_subid, 'static','image_data','recon_resample.nii.gz')).get_fdata()
#         mae, mse, rmse, r_rmse, ssim = ff.compare(motion[:,:,5:], static[:,:,5:], cutoff_low = -100)
#         print('mae: ', mae, ' rmse: ', rmse, ' ssim: ', ssim)

        
        
# #         # # monitor the process
# #         # max_value = np.max(recon_nb_image)
# #         # min_value = np.min(recon_nb_image)
# #         # print('max: ', max_value, 'min: ', min_value)
    
# #         # # check_list.append([patient_id, patient_subid,random_i, max_value, min_value, with_double_skull])
# #         # # df = pd.DataFrame(check_list, columns = ['patient_id', 'patient_subid','random' ,'max', 'min', 'with_double_skull'])
# #         # # df.to_excel(os.path.join(main_save_folder,'check_list_static.xlsx'))

    
