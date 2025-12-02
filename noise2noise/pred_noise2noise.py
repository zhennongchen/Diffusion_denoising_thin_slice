import sys
sys.path.append('/host/d/Github')
import os
import torch
import numpy as np
import nibabel as nb
import Diffusion_denoising_thin_slice.noise2noise.model as noise2noise
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator as Generator


#######################
trial_name = 'noise2noise_simple'
study_folder = '/host/d/projects/denoising/models'
epoch = 75
trained_model_filename = os.path.join(study_folder,trial_name, 'models/model-' + str(epoch)+ '.pt')
save_folder = os.path.join(study_folder, trial_name, 'pred_images_input_both'); os.makedirs(save_folder, exist_ok=True)

image_size = [512,512]

histogram_equalization = False
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 1000
#######################
build_sheet =  Build_list.Build(os.path.join('/host/d/Data/low_dose_CT/Patient_lists/mayo_low_dose_CT_gaussian_simulation_v2.xlsx'))
batch_list, patient_id_list, random_num_list, noise_file_all_list, noise_file_odd_list, noise_file_even_list, ground_truth_file_list, slice_num_list = build_sheet.__build__(batch_list = ['test'])
print('total cases:', patient_id_list.shape[0])
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 1)
print('total number:', n.shape[0])


# build model
model = noise2noise.Unet(
    problem_dimension = '2D',  # '2D' or '3D'
  
    input_channels = 1,
    out_channels = 1,  
    initial_dim = 16,  # initial feature dimension after first conv layer
    dim_mults = (2,4,8,16),
    groups = 8,
      
    attn_dim_head = 32,
    attn_heads = 4,
    full_attn_paths = (None, None, None, None), # these are for downsampling and upsampling paths
    full_attn_bottleneck = None, # this is for the middle bottleneck layer
    act = 'ReLU',
)

# main
G = Generator.Dataset_2D
for i in range(0,n.shape[0]):
    patient_id = patient_id_list[n[i]]
    random_num = random_num_list[n[i]]
    noise_file_all = noise_file_all_list[n[i]]
    noise_file_odd = noise_file_odd_list[n[i]]
    noise_file_even = noise_file_even_list[n[i]]
    gt_file = ground_truth_file_list[n[i]]

    # here we only use noise odd as condition
    condition_files = [noise_file_odd, noise_file_even]  # can be one or two condition files
    if len(condition_files) == 2:
        condition_names = ['odd','even']

    print(i,patient_id, random_num)

    # get the condition image (noise odd)
    affine = nb.load(condition_files[0]).affine
    condition_img = nb.load(condition_files[0]).get_fdata()[:,:,100:200]
    slice_num = condition_img.shape[2]
    print('slice num:', slice_num)

    # get ground truth image
    gt_img = nb.load(gt_file).get_fdata()

    # make folders
    save_folder_case = os.path.join(save_folder, patient_id, 'random_' + str(random_num), 'epoch' + str(epoch))
    ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, 'random_' + str(random_num)), save_folder_case])

    if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
        print('already done')
        continue
            
    for condition_i in range(0,len(condition_files)):

        condition_file = condition_files[condition_i]
        print('condition file:', condition_file)

        # generator
        generator = G(
            supervision = 'unsupervised',

            img_list = np.array([condition_file]), # this is a dummy, we do not use it
            condition_list = np.array([condition_file]),
            image_size = image_size,

            num_slices_per_image = 100,#slice_num, 
            random_pick_slice = False,
            slice_range = [100,200],# None,
            
            histogram_equalization = histogram_equalization,
            bins = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_lowdoseCT.npy'),
            bins_mapped = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_mapped_lowdoseCT.npy'),
            background_cutoff = background_cutoff,
            maximum_cutoff = maximum_cutoff,
            normalize_factor = normalize_factor,)

        # sample:
        sampler = noise2noise.Sampler(model,generator,batch_size = 1, image_size = image_size)

        pred_img = sampler.sample_2D(trained_model_filename, condition_img)
        print(pred_img.shape)
    
        # save
        if len(condition_files) == 1:
            nb.save(nb.Nifti1Image(pred_img, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))
        else:
            nb.save(nb.Nifti1Image(pred_img, affine), os.path.join(save_folder_case, 'pred_img_' + condition_names[condition_i] + '.nii.gz'))

    if len(condition_files) == 2:
        pred_img_final = np.zeros([len(condition_files), pred_img.shape[0], pred_img.shape[1], pred_img.shape[2]])
        for condition_i in range(0,len(condition_files)):
            pred_img_final[condition_i,:,:,:] = nb.load(os.path.join(save_folder_case, 'pred_img_' + condition_names[condition_i] + '.nii.gz')).get_fdata()
        # average the two conditions
        pred_img_final = np.mean(pred_img_final, axis = 0)
        assert pred_img_final.shape == pred_img.shape
        nb.save(nb.Nifti1Image(pred_img_final, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))

    # save condition
    # nb.save(nb.Nifti1Image(condition_img, affine),  os.path.join(save_folder_case, 'condition_img.nii.gz'))