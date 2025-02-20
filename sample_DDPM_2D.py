import sys
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator as Generator

###########
trial_name = 'supervised_possion_2D'
problem_dimension = '2D'
epoch = 50
trained_model_filename = os.path.join('/mnt/camca_NAS/denoising/models', trial_name, 'models/model-' + str(epoch)+ '.pt')
save_folder = os.path.join('/mnt/camca_NAS/denoising/models', trial_name, 'pred_images'); os.makedirs(save_folder, exist_ok=True)

image_size = [512,512]

objective = 'pred_x0'
timesteps = 1000
sampling_timesteps = 1000
eta = 0. # usually use 1.

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'
clip_range = [-1,1]

###########
build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_possion.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [0,1,2,3]) 
x0_list = x0_list[0:1]; condition_list = condition_list[0:1]

model = ddpm.Unet(
    problem_dimension = problem_dimension,
    init_dim = 64,
    out_dim = 1,
    channels = 1, 
    conditional_diffusion = True,
    condition_channels = 1, 
    downsample_list = (True, True, True, False),
    upsample_list = (True, True, True, False),
    full_attn = (None, None, False, True),)

diffusion_model = ddpm.GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = timesteps,           # number of steps
    sampling_timesteps = sampling_timesteps,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    ddim_sampling_eta = eta,
    force_ddim = False,
    objective = objective,
    clip_or_not = True, 
    clip_range = clip_range, )

for i in range(0,x0_list.shape[0]):
    patient_id = patient_id_list[i]
    patient_subid = patient_subid_list[i]
    random_num = random_num_list[i]
    x0_file = x0_list[i]
    condition_file = condition_list[i]

    print(i,patient_id, patient_subid, random_num)

    # get the ground truth image
    gt_img = nb.load(x0_file)
    affine = gt_img.affine; gt_img = gt_img.get_fdata()[:,:,50:52]

    # get the condition image
    condition_img = nb.load(condition_file).get_fdata()[:,:,50:52]

    # make folders
    ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, patient_subid), os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num))])
    save_folder_case = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'epoch' + str(epoch)); os.makedirs(save_folder_case, exist_ok=True)

    # generator
    generator = Generator.Dataset_2D(
        img_list = np.array([x0_file]),
        condition_list = np.array([condition_file]),
        image_size = image_size,

        num_slices_per_image = 2,#gt_img.shape[-1],
        random_pick_slice = False,
        slice_range = [50,52],

        histogram_equalization = histogram_equalization,
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,)

    # sample:
    sampler = ddpm.Sampler(diffusion_model,generator,batch_size = 1)

    pred_img = sampler.sample_2D(trained_model_filename, gt_img)
  
    # save
    nb.save(nb.Nifti1Image(pred_img, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))
    nb.save(nb.Nifti1Image(gt_img, affine), os.path.join(save_folder_case, 'gt_img.nii.gz'))
    nb.save(nb.Nifti1Image(condition_img, affine), os.path.join(save_folder_case, 'condition_img.nii.gz'))