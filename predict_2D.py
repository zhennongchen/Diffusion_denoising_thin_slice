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

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'
clip_range = [-1,1]

###########
build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian_local.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 
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
    sampling_timesteps = 1000,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    ddim_sampling_eta = 1.,
    force_ddim = False,
    auto_normalize=False,
    objective = objective,
    clip_or_not = True, 
    clip_range = clip_range, )

for i in list(range(0, 3+ 1, 3)):#range(0,1):#x0_list.shape[0]):
    patient_id = patient_id_list[i]
    patient_subid = patient_subid_list[i]
    random_num = random_num_list[i]
    x0_file = x0_list[i]
    condition_file = condition_list[i]

    print(i,patient_id, patient_subid, random_num)

    # get the ground truth image
    gt_img = nb.load(x0_file)
    affine = gt_img.affine; gt_img = gt_img.get_fdata()[:,:,40:60]

    # get the condition image
    condition_img = nb.load(condition_file).get_fdata()[:,:,40:60]

    # make folders
    ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, patient_subid), os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num))])
    # save_folder_case = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'epoch' + str(epoch)+'_5'); os.makedirs(save_folder_case, exist_ok=True)

    # generator
    # generator = Generator.Dataset_2D(
    #     supervision = supervision,

    #     img_list = np.array([x0_file]),
    #     condition_list = np.array([condition_file]),
    #     image_size = image_size,

    #     num_slices_per_image = 20, 
    #     random_pick_slice = False,
    #     slice_range = [40,60],

    #     histogram_equalization = histogram_equalization,
    #     background_cutoff = background_cutoff,
    #     maximum_cutoff = maximum_cutoff,
    #     normalize_factor = normalize_factor,)

    # # sample:
    # sampler = ddpm.Sampler(diffusion_model,generator,batch_size = 1)

    # pred_img = sampler.sample_2D(trained_model_filename, gt_img)
    # print(pred_img.shape)

    # # if supervision == 'unsupervised':
    # #     pred_img_final = np.zeros(gt_img.shape)
    # #     pred_img_final[:,:,0] = gt_img[:,:,0]
    # #     pred_img_final[:,:,1:pred_img_final.shape[-1]-1] = pred_img[:,:,0:pred_img_final.shape[-1]-2]
    # #     pred_img_final[:,:,-1] = gt_img[:,:,-1]
    # # else:
    # #     pred_img_final = pred_img

    # pred_img_final = pred_img
  
    # # save
    # nb.save(nb.Nifti1Image(pred_img_final, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))
    # nb.save(nb.Nifti1Image(gt_img, affine), os.path.join(save_folder_case, 'gt_img.nii.gz'))
    # nb.save(nb.Nifti1Image(condition_img, affine), os.path.join(save_folder_case, 'condition_img.nii.gz'))

    # 
    save_folder_case = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'epoch' + str(epoch)+'final'); os.makedirs(save_folder_case, exist_ok=True)
    made_predicts = ff.find_all_target_files(['epoch' + str(epoch)+'_*'], os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num)))
    print(made_predicts)

    predicts_final = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2], len(made_predicts)))
    for j in range(len(made_predicts)):
        predicts_final[:,:,:,j] = nb.load(os.path.join(made_predicts[j],'pred_img.nii.gz')).get_fdata()
    # average across last axis
    predicts_final = np.mean(predicts_final, axis = -1)
    nb.save(nb.Nifti1Image(predicts_final, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))
