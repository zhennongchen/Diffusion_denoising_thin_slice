import sys
sys.path.append('/host/d/Github')
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM as edm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator as Generator

###########
trial_name = 'unsupervised_gaussian'
problem_dimension = '2D'
supervision = 'supervised' if trial_name[0:2] == 'su' else 'unsupervised'; print('supervision:', supervision)

study_folder = '/host/d/projects/denoising/models'
epoch = 100
trained_model_filename = os.path.join(study_folder,trial_name, 'models/model-' + str(epoch)+ '.pt')
save_folder = os.path.join(study_folder, trial_name, 'pred_images'); os.makedirs(save_folder, exist_ok=True)

# bias 
beta = 0

# model condition 
condition_channel = 1

image_size = [512,512] 
objective = 'pred_x0'
sampling_timesteps = 100

histogram_equalization = False
assert not histogram_equalization, "histogram equalization not needed for this experiment"
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'
clip_range = [-1,1]

do_pred_or_avg = 'pred'

###########
build_sheet =  Build_list.Build(os.path.join('/host/d/Data/low_dose_CT/Patient_lists/mayo_low_dose_CT_gaussian_simulation_v1.xlsx'))
batch_list, patient_id_list, random_num_list, noise_file_odd_list, noise_file_even_list, ground_truth_file_list, slice_num_list = build_sheet.__build__(batch_list = ['test'])
print('total cases:', patient_id_list.shape[0])
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 2)
print('total number:', n.shape[0])

model = ddpm.Unet(
    problem_dimension = problem_dimension,
    init_dim = 64,
    out_dim = 1,
    channels = 1, 
    conditional_diffusion = True,
    condition_channels = condition_channel,

    downsample_list = (True, True, True, False),
    upsample_list = (True, True, True, False),
    full_attn = (None, None, False, True),)


diffusion_model = ddpm.GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = 1000,           # number of steps
    sampling_timesteps = sampling_timesteps,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    ddim_sampling_eta = 1.,
    force_ddim = False,
    auto_normalize=False,
    objective = objective,
    clip_or_not = True, 
    clip_range = clip_range, )

for i in range(0, n.shape[0]):
    patient_id = patient_id_list[n[i]]
    random_num = random_num_list[n[i]]
    noise_file_odd = noise_file_odd_list[n[i]]
    noise_file_even = noise_file_even_list[n[i]]
    gt_file = ground_truth_file_list[n[i]]

    # here we only use noise odd as condition
    condition_file = noise_file_odd

    print(i,patient_id, random_num)

    # get the condition image (noise odd)
    affine = nb.load(condition_file).affine
    condition_img = nb.load(noise_file_odd).get_fdata()
    slice_num = condition_img.shape[2]
    print('slice num:', slice_num)

    if do_pred_or_avg == 'pred':
        for iteration in range(1,21):
            print('iteration:', iteration)

            # make folders
            save_folder_case = os.path.join(save_folder, patient_id, 'random_' + str(random_num), 'epoch' + str(epoch)+'_' + str(iteration))
            ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, 'random_' + str(random_num)), save_folder_case])

            if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                print('already done')
                continue

            # generator
            generator = Generator.Dataset_2D(
                supervision = supervision,

                img_list = np.array([condition_file]), # this is a dummy, we do not use it
                condition_list = np.array([condition_file]),
                image_size = image_size,

                num_slices_per_image = slice_num, 
                random_pick_slice = False,
                slice_range = None,

                histogram_equalization = histogram_equalization,
                bins = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins.npy'),
                bins_mapped = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_mapped.npy'),
                background_cutoff = background_cutoff,
                maximum_cutoff = maximum_cutoff,
                normalize_factor = normalize_factor,)

            # sample:
            sampler = ddpm.Sampler(diffusion_model,generator,batch_size = 1)

            pred_img = sampler.sample_2D(trained_model_filename, condition_img)
            print(pred_img.shape)

            pred_img_final = pred_img
    
            # save
            nb.save(nb.Nifti1Image(pred_img_final, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))
       

    if do_pred_or_avg == 'avg':

        save_folder_avg = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'epoch' + str(epoch)+'avg'); os.makedirs(save_folder_avg, exist_ok=True)

        if os.path.isfile(os.path.join(save_folder_avg, 'pred_img_scans20.nii.gz')):
            print('already done')
            continue
        
        made_predicts = ff.sort_timeframe(ff.find_all_target_files(['epoch' + str(epoch)+'_*'], os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num))),0,'_','/')
        if len(made_predicts) == 0:
            print('skip, no made predicts')
            continue
        total_predicts = 0
        for jj in range(len(made_predicts)):
            total_predicts += os.path.isfile(os.path.join(made_predicts[jj],'pred_img.nii.gz'))
        print('total made predicts:', total_predicts)
        if total_predicts != 20:
            print('skip, not enough predicts')
            continue

        loaded_data = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2], total_predicts))
        for j in range(total_predicts):
            loaded_data[:,:,:,j] = nb.load(os.path.join(made_predicts[j],'pred_img.nii.gz')).get_fdata()

        for avg_num in [10,20]:#[2,4,6,8,10,12,14,16,18,20]:#range(1,total_predicts+1):
            print('avg_num:', avg_num)
            predicts_avg = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2], avg_num))
            print('predict_num:', avg_num)
            for j in range(avg_num):
                print('file:', made_predicts[j])
                predicts_avg[:,:,:,j] = loaded_data[:,:,:,j]
            # average across last axis
            predicts_avg = np.mean(predicts_avg, axis = -1)
            nb.save(nb.Nifti1Image(predicts_avg, affine), os.path.join(save_folder_avg, 'pred_img_scans' + str(avg_num) + '.nii.gz'))