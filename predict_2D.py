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

###########
trial_name = 'unsupervised_DDPM_gaussian_2D'
problem_dimension = '2D'
supervision = 'supervised' if trial_name[0:2] == 'su' else 'unsupervised'; print('supervision:', supervision)
epoch = 70
trained_model_filename = os.path.join('/mnt/camca_NAS/denoising/models', trial_name, 'models/model-' + str(epoch)+ '.pt')
save_folder = os.path.join('/mnt/camca_NAS/denoising/models', trial_name, 'pred_images'); os.makedirs(save_folder, exist_ok=True)

image_size = [512,512]

objective = 'pred_x0'
sampling_timesteps = 200

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'
clip_range = [-1,1]

do_pred_or_avg = 'avg'

###########
build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 3)
# x0_list = x0_list[0:1]; condition_list = condition_list[0:1]

model = ddpm.Unet(
    problem_dimension = problem_dimension,
    init_dim = 64,
    out_dim = 1,
    channels = 1, 
    conditional_diffusion = True,
    condition_channels = 1 if supervision == 'supervised' else 2,

    downsample_list = (True, True, True, False),
    upsample_list = (True, True, True, False),
    full_attn = (None, None, False, True),)

# diffusion_model = edm.EDM(
#     model,
#     image_size = image_size,
#     num_sample_steps = 100,
#     clip_or_not = True,
#     clip_range = clip_range,)

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

for i in range(0,n.shape[0]):
    patient_id = patient_id_list[n[i]]
    patient_subid = patient_subid_list[n[i]]
    random_num = random_num_list[n[i]]
    x0_file = x0_list[n[i]]
    condition_file = condition_list[n[i]]

    print(i,patient_id, patient_subid, random_num)


    # get the ground truth image
    gt_img = nb.load(x0_file)
    affine = gt_img.affine; gt_img = gt_img.get_fdata()[:,:,30:80]

    # get the condition image
    condition_img = nb.load(condition_file).get_fdata()[:,:,30:80]

    for iteration in range(0,1):
        print('iteration:', iteration)

        # make folders
        if do_pred_or_avg == 'pred':
            ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, patient_subid), os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num))])
            save_folder_case = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'epoch' + str(epoch)+'_'+str(iteration)); os.makedirs(save_folder_case, exist_ok=True)

            if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                print('already done')
                continue

            # generator
            generator = Generator.Dataset_2D(
                supervision = supervision,

                img_list = np.array([x0_file]),
                condition_list = np.array([condition_file]),
                image_size = image_size,

                num_slices_per_image = 50, 
                random_pick_slice = False,
                slice_range = [30,80],

                histogram_equalization = histogram_equalization,
                background_cutoff = background_cutoff,
                maximum_cutoff = maximum_cutoff,
                normalize_factor = normalize_factor,)

            # sample:
            sampler = ddpm.Sampler(diffusion_model,generator,batch_size = 1)

            pred_img = sampler.sample_2D(trained_model_filename, gt_img)
            print(pred_img.shape)

            pred_img_final = pred_img
    
            # save
            nb.save(nb.Nifti1Image(pred_img_final, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))

            if iteration == 1:
                nb.save(nb.Nifti1Image(gt_img, affine), os.path.join(save_folder_case, 'gt_img.nii.gz'))
                nb.save(nb.Nifti1Image(condition_img, affine), os.path.join(save_folder_case, 'condition_img.nii.gz'))
                # # also save the possion noise image
                if 'possion' in trial_name:
                    possion_file = os.path.join('/workspace/Documents/Data/denoising/simulation', patient_id, patient_subid, 'possion_random_' + str(random_num), 'recon.nii.gz')
                    possion_img = nb.load(possion_file).get_fdata()[:,:,30:80]
                    nb.save(nb.Nifti1Image(possion_img, affine), os.path.join(save_folder_case, 'possion_img.nii.gz'))

        # 
        if do_pred_or_avg == 'avg':

            save_folder_avg = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'epoch' + str(epoch)+'avg'); os.makedirs(save_folder_avg, exist_ok=True)
            
            made_predicts = ff.sort_timeframe(ff.find_all_target_files(['epoch' + str(epoch)+'_*'], os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num))),0,'_','/')
            print(made_predicts)
            total_predicts = len(made_predicts)

            loaded_data = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2], total_predicts))
            for j in range(total_predicts):
                loaded_data[:,:,:,j] = nb.load(os.path.join(made_predicts[j],'pred_img.nii.gz')).get_fdata()

            for avg_num in range(1,total_predicts+1):
                predicts_avg = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2], avg_num))
                print('predict_num:', avg_num)
                for j in range(avg_num):
                    print('file:', made_predicts[j])
                    predicts_avg[:,:,:,j] = loaded_data[:,:,:,j]
                # average across last axis
                predicts_avg = np.mean(predicts_avg, axis = -1)
                nb.save(nb.Nifti1Image(predicts_avg, affine), os.path.join(save_folder_avg, 'pred_img_scans' + str(avg_num) + '.nii.gz'))