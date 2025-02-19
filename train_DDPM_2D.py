import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator as Generator

trial_name = 'supervised_possion_2D'
problem_dimension = '2D'
pre_trained_model = None#os.path.join(cg.diffusion_dir,'models','portable_DDPM_patch_3Dmotion_hist_v1', 'models', 'model-9.pt')
start_step = 0
image_size = [512,512]

objective = 'pred_x0'
timesteps = 1000

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'

###########################
# define train
build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_possion.xlsx'))
_,_,_,_, condition_list_train, x0_list_train = build_sheet.__build__(batch_list = [0,1,2,3]) 
x0_list_train = x0_list_train[0:1]; condition_list_train = condition_list_train[0:1]
 
# define val
_,_,_,_, condition_list_val, x0_list_val = build_sheet.__build__(batch_list = [0,1,2,3])
x0_list_val = x0_list_val[0:1]; condition_list_val = condition_list_val[0:1]

print('train:', x0_list_train.shape, condition_list_train.shape, 'val:', x0_list_val.shape, condition_list_val.shape)
print(x0_list_train[0:3], condition_list_train[0:3], x0_list_val[0:3], condition_list_val[0:3])

# define u-net and diffusion model
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
    sampling_timesteps = 250,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    objective = objective,
    clip_or_not = False,)

# generator definition
generator_train = Generator.Dataset_2D(
        img_list = x0_list_train,
        condition_list = condition_list_train,
        image_size = image_size,

        num_slices_per_image = 156,
        random_pick_slice = False,
        slice_range = None,

        histogram_equalization = histogram_equalization,
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,

        shuffle = True,
        augment = True,
        augment_frequency = 0.5,)


generator_val = Generator.Dataset_2D(
        img_list = x0_list_val,
        condition_list = condition_list_val,
        image_size = image_size,

        num_slices_per_image = 50,
        random_pick_slice = False,
        slice_range = [40,110],

        histogram_equalization = histogram_equalization,
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,)

# start to train
trainer = ddpm.Trainer(
    diffusion_model= diffusion_model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = 1,
    
    accum_iter = 20,
    train_num_steps = 20000, # total training epochs
    results_folder = os.path.join('/mnt/camca_NAS/denoising/models', trial_name, 'models'),
   
    train_lr = 1e-4,
    train_lr_decay_every = 2000, 
    save_models_every = 50,
    validation_every = 1000000000000,)


trainer.train(pre_trained_model=pre_trained_model, start_step= start_step )