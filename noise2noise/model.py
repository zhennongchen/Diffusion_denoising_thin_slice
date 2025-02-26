import math
import copy
import os
import pandas as pd
import numpy as np
import nibabel as nb
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from skimage.measure import block_reduce

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from Diffusion_denoising_thin_slice.noise2noise.attend import Attend
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Data_processing as Data_processing


# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val): 
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def Upsample2D(dim, dim_out = None, upsample_factor = (2,2)):
    return nn.Sequential(
        nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Upsample3D(dim, dim_out = None, upsample_factor = (2,2,1)):
    return nn.Sequential(
        nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample2D(dim, dim_out = None):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
        nn.Conv2d(dim, default(dim_out, dim), 1)
    )

def Downsample3D(dim, dim_out = None):
    return nn.Sequential(
        nn.MaxPool3d(kernel_size=(2,2, 1), stride=(2,2, 1), padding=0),
        nn.Conv3d(dim, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    '''RMSNorm applies channel-wise normalization to the input tensor, 
    scales the normalized values using the learnable parameter g, 
    and then further scales the result by the square root of the number of input channels. '''
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # learnable

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class RMSNorm3D(nn.Module):
    '''RMSNorm applies channel-wise normalization to the input tensor, 
    scales the normalized values using the learnable parameter g, 
    and then further scales the result by the square root of the number of input channels. '''
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1 , 1)) # learnable

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# building block modules
    
class ConvBlock2D(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups = 8, dilation = None, act = 'ReLU'):
        super().__init__()
        if dilation == None:
            self.conv = nn.Conv2d(dim, dim_out, 3, padding = 1)
        else:
            self.conv = nn.Conv2d(dim, dim_out, 3, padding = dilation, dilation = dilation)

        self.norm = nn.GroupNorm(groups, dim_out)  

        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError('activation function not supported')
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ConvBlock3D(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups = 8, dilation = None, act = 'ReLU'):
        super().__init__()
        if dilation == None:
            self.conv = nn.Conv3d(dim, dim_out, 3, padding = 1)
        else:
            self.conv = nn.Conv3d(dim, dim_out, 3, padding = dilation, dilation = dilation)
        self.norm = nn.GroupNorm(groups, dim_out) 
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError('activation function not supported')

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# Attention:
class LinearAttention2D(nn.Module): # input dimension is dim, same dimension for input and output
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)  # split input into q k v evenly
        # here each q, k ,v has the dim = [b, hidden_dim, h, w]

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        # here each q, k ,v has the dim = [b, heads, hidden_dim, h*w]

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # matrix multiplication
        # k*v:  [b, heads, hidden_dim, h*w] mul [b, heads, hidden_dim, h*w] -> [b, heads, hidden_dim, hidden_dim]

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # context*q: [b, heads, hidden_dim, hidden_dim] mul [b, heads, hidden_dim, h*w] -> [b, heads, hidden_dim, h*w]

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        # out: [b, heads, hidden_dim, h*w] -> [b, heads*hidden_dim, h, w]
        return self.to_out(out)
    

class Attention2D(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim) 
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # here each q, k ,v has the dim = [b, hidden_dim, h, w]

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)
        # here each q, k ,v has the dim = [b, heads, h*w, hidden_dim]

        out = self.attend(q, k, v)
        # first q*k: [b, heads, h*w, hidden_dim] mul [b, heads, h*w, hidden_dim] -> [b, heads, h*w, h*w]   (einsum(f"b h i d, b h j d -> b h i j", q, k) * scale)
        # second *v: [b, heads, h*w, h*w] mul [b, heads, h*w, hidden_dim] -> [b, heads, h*w, hidden_dim]  (einsum(f"b h i j, b h j d -> b h i d", attn, v))

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)
    
    
class LinearAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm3D(dim)
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv3d(hidden_dim, dim, 1),
            RMSNorm3D(dim)
        )

    def forward(self, x):
        b, c, h, w, d = x.shape  # Added dimension 'd' for depth

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)  # split input into q k v evenly
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h=self.heads), qkv)  # h = head, c = dim_head

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # matrix multiplication

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h = self.heads, x = h, y = w, z = d)
        return self.to_out(out)


class Attention3D(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm3D(dim)
        self.attend = Attend(flash = False)

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape  # Added dimension 'd' for depth

        x = self.norm(x) 

        qkv = self.to_qkv(x).chunk(3, dim=1)  # split input into q k v evenly
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h (x y z) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z = d)
        return self.to_out(out)
    

class ResnetBlock2D(nn.Module): 
    # conv + conv + attention + residual
    def __init__(self, dim, dim_out, groups = 8, use_full_attention = None, attn_head = 4, attn_dim_head = 32, act = 'ReLU'):
        '''usee which attention: 'Full' or 'Linear'''
        super().__init__()
    
        self.block1 = ConvBlock2D(dim, dim_out, groups = groups, act = act)
        self.block2 = ConvBlock2D(dim_out, dim_out, groups = groups , act = act)
        
        if use_full_attention == True:
            self.attention = Attention2D(dim_out, heads = attn_head, dim_head = attn_dim_head)
        elif use_full_attention == False:
            self.attention = LinearAttention2D(dim_out, heads = attn_head, dim_head = attn_dim_head)
        else:
            self.attention = nn.Identity()

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):

        h = self.block1(x) 

        h = self.block2(h)

        h = self.attention(h)

        return h + self.res_conv(x)
    
    
class ResnetBlock3D(nn.Module): 
    # conv + conv + attention + residual
    def __init__(self, dim, dim_out, groups = 8, use_full_attention = None, attn_head = 4, attn_dim_head = 32 , act = 'ReLU'):
        '''usee which attention: 'Full' or 'Linear'''
        super().__init__()
    
        self.block1 = ConvBlock3D(dim, dim_out, groups = groups, act = act)
        self.block2 = ConvBlock3D(dim_out, dim_out, groups = groups, act = act)
        
        if use_full_attention == True:
            self.attention = Attention3D(dim_out, heads = attn_head, dim_head = attn_dim_head)
        elif use_full_attention == False:
            self.attention = LinearAttention3D(dim_out, heads = attn_head, dim_head = attn_dim_head)
        else:
            self.attention = nn.Identity()

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):

        h = self.block1(x) 

        h = self.block2(h)

        h = self.attention(h)

        return h + self.res_conv(x)
    
    
class Unet2D(nn.Module):
    def __init__(
        self,
        init_dim = 16,
        channels = 1,

        out_dim = None,
        dim_mults = (2,4,8,16),
        self_condition = False,   # use the prediction from the previous iteration as the condition of next iteration
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (None, None, None, True),
        act = 'ReLU',
    ):
        super().__init__()
    
        self.channels = channels
        input_channels = channels

        self.init_conv = nn.Conv2d(input_channels, init_dim, 3, padding = 1) # if want input and output to have same dimension, Kernel size to any odd number (e.g., 3, 5, 7, etc.). Padding to (kernel size - 1) / 2.

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]  # if initi_dim = 16, then [16, 32, 64, 128, 256]

        in_out = list(zip(dims[:-1], dims[1:])) 
        print('in out is : ', in_out)
        # [(16,32), (32,64), (64,128), (128,256)]. Each tuple in in_out represents a pair of input and output dimensions for different stages in a neural network 

        # attention
        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # 4

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            # print(' in downsampling path, ind is: ', ind, ' dim_in is: ', dim_in, ' dim_out is: ', dim_out, ' layer_full_attn is: ', layer_full_attn, ' layer_attn_heads is: ', layer_attn_heads, ' layer_attn_dim_head is: ', layer_attn_dim_head)
            is_last = ind >= (num_resolutions - 1)

            # in each downsample stage, 
            # we have a resnetblock and then downsampling layer (downsample x and y by 2, then increase the feature number by 2)
            self.downs.append(nn.ModuleList([
                ResnetBlock2D(dim_in, dim_in, use_full_attention = layer_full_attn, attn_head = layer_attn_heads, attn_dim_head = layer_attn_dim_head, act = act),
                Downsample2D(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block = ResnetBlock2D(mid_dim, mid_dim, use_full_attention = True, attn_head = attn_heads[-1], attn_dim_head = attn_dim_head[-1], act = act)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            # print(' in upsampling path, ind is: ', ind, ' dim_in is: ', dim_in, ' dim_out is: ', dim_out, ' layer_full_attn is: ', layer_full_attn, ' layer_attn_heads is: ', layer_attn_heads, ' layer_attn_dim_head is: ', layer_attn_dim_head)
            is_last = ind == (len(in_out) - 1)
          
            # in each upsample stage,
            # we have a resnetblock and then upsampling layer (upsample x and y by 2, then decrease the feature number by 2)
            self.ups.append(nn.ModuleList([
                ResnetBlock2D(dim_out + dim_in, dim_out, use_full_attention = layer_full_attn, attn_head = layer_attn_heads, attn_dim_head = layer_attn_dim_head, act = act),
                Upsample2D(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 5, padding = 2)  
            ]))

        self.out_dim = out_dim if out_dim is not None else channels
        self.final_res_block = ResnetBlock2D(init_dim * 2, init_dim, use_full_attention = None, attn_head = attn_heads[0], attn_dim_head = attn_dim_head[0], act = act)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)  # output channel is initial channel number

    def forward(self, x):

        x = self.init_conv(x)
        print('initial x shape is: ', x.shape)
        x_init = x.clone()

        h = []
        for block, downsample in self.downs:
            x = block(x)
            h.append(x)

            x = downsample(x)
        
        x = self.mid_block(x)
        print('middle x shape is: ', x.shape)
        
        for block, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)   # h.pop() is the output of the corresponding downsample stage
            x = block(x)
            x = upsample(x)

        x = torch.cat((x, x_init), dim = 1)

        x = self.final_res_block(x)
        final_image = self.final_conv(x)
        print('final image shape is: ', final_image.shape)
      
        return final_image


class Trainer(object):
    def __init__(
        self,
        model,
        generator_train,
        generator_val,
        train_batch_size,

        *,
        accum_iter = 10, # gradient accumulation steps
        train_num_steps = 10000, # total training epochs
        results_folder = None,
        train_lr = 1e-4,
        train_lr_decay_every = 100, 
        save_models_every = 1,
        validation_every = 1,
        
        ema_update_every = 10,
        ema_decay = 0.95,
        adam_betas = (0.9, 0.99),

        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,

        max_grad_norm = 1.,
         
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = model  
        # self.specify_loss = specify_loss
        # if self.specify_loss == 'mse':
        #     self.loss_function = F.mse_loss
        # elif self.specify_loss == 'mae':
        #     self.loss_function = F.l1_loss
        # elif self.specify_loss == 'both':
        self.loss_function1 = F.mse_loss; self.loss_function2 = F.l1_loss

        self.channels = model.channels

        # sampling and training hyperparameters
        self.batch_size = train_batch_size
        self.accum_iter = accum_iter
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = generator_train
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl = self.accelerator.prepare(dl)

        self.ds_val = generator_val
        dl_val = DataLoader(self.ds_val, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl_val = self.accelerator.prepare(dl_val)

        # optimizer
        self.opt = Adam(model.parameters(), lr = train_lr, betas = adam_betas)
        self.scheduler = StepLR(self.opt, step_size = 1, gamma=0.95)
        self.train_lr_decay_every = train_lr_decay_every
        self.save_model_every = save_models_every
        self.validation_every = validation_every

        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = results_folder
    
        ff.make_folder([self.results_folder])

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, stepNum):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'decay_steps': self.scheduler.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        
        torch.save(data, os.path.join(self.results_folder, 'model-' + str(stepNum) + '.pt'))

    def load_model(self, trained_model_filename):

        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(trained_model_filename, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        self.scheduler.load_state_dict(data['decay_steps'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def train(self, pre_trained_model = None ,start_step = None):

        accelerator = self.accelerator
        device = accelerator.device

        # load pre-trained
        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
            print('model loaded from ', pre_trained_model)

        if start_step is not None:
            self.step = start_step

        training_log = []
        val_loss = np.inf
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            
            while self.step < self.train_num_steps:
                print('training epoch: ', self.step + 1)
                print('learning rate: ', self.scheduler.get_last_lr()[0])

                average_loss = []
                count = 1
                # load data
                for batch in self.dl:
                    if count == 1 or count % self.accum_iter == 1 or count == len(self.dl) - 1 or count == len(self.dl):
                        self.opt.zero_grad()
         
                    # load data
                    batch_input, batch_gt = batch 
                    data_input = batch_input.to(device)
                    data_gt = batch_gt.to(device)

                    with self.accelerator.autocast():
                        output = self.model(data_input)
                        
                        # loss
                        loss = self.loss_function1(output, data_gt) + self.loss_function2(output, data_gt)
                        
                    # accumulate the gradient, typically used when batch size is small
                    if count % self.accum_iter == 0 or count == len(self.dl) - 1 or count == len(self.dl):
                        self.accelerator.backward(loss)
                        accelerator.wait_for_everyone()
                        accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.opt.step()

                    count += 1
                    average_loss.append(loss.item())

                   
                average_loss = sum(average_loss) / len(average_loss)
                pbar.set_description(f'average loss: {average_loss:.4f}')
               
                accelerator.wait_for_everyone()

                self.step += 1

                # save the model
                if self.step !=0 and divisible_by(self.step, self.save_model_every):
                   self.save(self.step)
                
                if self.step !=0 and divisible_by(self.step, self.train_lr_decay_every):
                    self.scheduler.step()
                    
                self.ema.update()

                # do the validation if necessary
                if self.step !=0 and divisible_by(self.step, self.validation_every):
                    print('validation at step: ', self.step)
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = []
                        for batch in self.dl_val:
                            batch_input, batch_gt = batch 
                            data_input = batch_input.to(device)
                            data_gt = batch_gt.to(device)
                            with self.accelerator.autocast():
                                output = self.model(data_input)
                                loss = self.loss_function1(output, data_gt) + self.loss_function2(output, data_gt)
                            
                            val_loss.append(loss.item())
                        val_loss = sum(val_loss) / len(val_loss)
                        print('validation loss: ', val_loss)
                    self.model.train(True)

                # save the training log
                training_log.append([self.step,self.scheduler.get_last_lr()[0],average_loss,val_loss])
                df = pd.DataFrame(training_log,columns = ['iteration','learning_rate','training_loss','validation_loss'])
                log_folder = os.path.join(os.path.dirname(self.results_folder),'log');ff.make_folder([log_folder])
                df.to_excel(os.path.join(log_folder, 'training_log.xlsx'),index=False)
                        
                # at the end of each epoch, call on_epoch_end
                self.ds.on_epoch_end(); self.ds_val.on_epoch_end()

                pbar.update(1)

        accelerator.print('training complete')



# # Sampling class
# class Sampler(object):
#     def __init__(
#         self,
#         model,
#         generator,
#         batch_size,
#         image_size = None,
#         device = 'cuda',

#     ):
#         super().__init__()

#         # model
#         self.model = model  
#         if device == 'cuda':
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if device == 'cpu':
#             self.device = torch.device("cpu")

#         self.channels = model.channels
#         if image_size is None:
#             self.image_size = self.model.image_size
#         else:
#             self.image_size = image_size
#         self.batch_size = batch_size

#         # dataset and dataloader

#         self.generator = generator
#         dl = DataLoader(self.generator, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())

#         self.dl = dl
#         self.cycle_dl = cycle(dl)
 
#         # EMA:
#         self.ema = EMA(model)
#         self.ema.to(self.device)

#     def load_model(self, trained_model_filename):

#         data = torch.load(trained_model_filename, map_location=self.device)

#         self.model.load_state_dict(data['model'])

#         self.step = data['step']

#         self.ema.load_state_dict(data["ema"])


#     def sample_patches_w_trained_model(self, trained_model_filename, ground_truth_image_file, motion_image_file,  slice_range, save_folder, patch_origin_list , patch_size, count_matrix,  save_gt_motion = None, portable_CT = False):
        
#         background_cutoff = self.generator.background_cutoff
#         maximum_cutoff = self.generator.maximum_cutoff
#         normalize_factor = self.generator.normalize_factor

#         self.load_model(trained_model_filename) 
        
#         device = self.device

#         self.ema.ema_model.eval()
#         # check whether model is on GPU:
#         print('model device: ', next(self.ema.ema_model.parameters()).device)

#         final_image = np.zeros(self.image_size)

#         gt = nb.load(ground_truth_image_file)
#         gt_img = gt.get_fdata()
#         if portable_CT == False:
#             gt_img = gt_img[:,:,slice_range[0]: slice_range[1]]
#         else:
#             gt_img = gt_img[:,:,slice_range[0] + 10: slice_range[1] + 10]
   
#         gt_img = Data_processing.cutoff_intensity(gt_img, cutoff_low = background_cutoff, cutoff_high = maximum_cutoff)
#         nb.save(nb.Nifti1Image(gt_img, gt.affine), os.path.join(save_folder , 'gt.nii.gz'))

#         # start to run
#         final_image = np.zeros(self.image_size)
#         with torch.inference_mode():
#             for k in range(0, len(patch_origin_list)):
#                 current_origin = patch_origin_list[k]

#                 datas = next(self.cycle_dl) 
#                 data_input = datas[1]
#                 print('data input shape: ', data_input.shape)

#                 data_input = data_input.to(torch.float32).to(device)

#                 pred_img = self.ema.ema_model(data_input)

#                 pred_img = pred_img.detach().cpu().numpy().squeeze() + data_input.detach().cpu().numpy().squeeze()
#                 pred_img = Data_processing.normalize_image(pred_img, normalize_factor = normalize_factor, image_max = maximum_cutoff, image_min = background_cutoff, invert = True)
#                 nb.save(nb.Nifti1Image(pred_img, gt.affine), os.path.join(save_folder, 'pred_origin_'+ str(current_origin[0]) + '_' + str(current_origin[1]) + '.nii.gz'))
#                 print(pred_img.shape)

#                 final_image[current_origin[0]: (current_origin[0] + patch_size), current_origin[1]: (current_origin[1] + patch_size), slice_range[0]: slice_range[1]] += pred_img

#         final_image = np.divide(final_image, count_matrix)
#         final_image = Data_processing.crop_or_pad(final_image, [gt_img.shape[0], gt_img.shape[1], self.image_size[-1]], value = np.min(gt_img))
#         final_image = Data_processing.correct_shift_caused_in_pad_crop_loop(final_image)
      
#         nb.save(nb.Nifti1Image(final_image, gt.affine), os.path.join(save_folder, 'pred.nii.gz'))

#         # save gt and motion
#         if save_gt_motion:
#             motion_img = nb.load(motion_image_file).get_fdata()
#             if portable_CT == False:
#                 motion_img = motion_img[:,:,slice_range[0]: slice_range[1]]
#             else:
#                 motion_img = motion_img[:,:,slice_range[0] + 10: slice_range[1] + 10]
#             motion_img = Data_processing.cutoff_intensity(motion_img, cutoff_low = background_cutoff, cutoff_high = maximum_cutoff)
#             motion_img_save = nb.Nifti1Image(motion_img, gt.affine)
#             nb.save(motion_img_save, os.path.join(save_folder, 'motion.nii.gz'))

#             nb.save(nb.Nifti1Image(gt_img - motion_img, gt.affine), os.path.join(save_folder , 'gt_delta.nii.gz'))

    
#     def sample_2D_w_trained_model(self, trained_model_filename, ground_truth_image_file, motion_image_file, save_file, slice_range, background_cutoff, maximum_cutoff, normalize_factor, save_gt_motion = None):
     
#         self.load_model(trained_model_filename)  
        
#         device = self.device

#         self.ema.ema_model.eval()
#         # check whether model is on GPU:
#         print('model device: ', next(self.ema.ema_model.parameters()).device)

#         pred_img_final_I = np.zeros([self.image_size[0], self.image_size[1],self.generator.num_slices])
#         pred_img_residual_I = np.zeros([self.image_size[0], self.image_size[1],self.generator.num_slices])

#         # define run iterations
#         if self.generator.num_slices % self.batch_size == 0:
#             run_iterations = self.generator.num_slices // self.batch_size
#         else:
#             run_iterations = self.generator.num_slices // self.batch_size + 1

#         # load gt
#         gt = nb.load(ground_truth_image_file)
#         gt_img = gt.get_fdata()
#         gt_img = gt_img[:,:,slice_range[0]: slice_range[1]]
#         gt_img = Data_processing.cutoff_intensity(gt_img, cutoff_low = background_cutoff, cutoff_high = maximum_cutoff)
#         nb.save(nb.Nifti1Image(gt_img, gt.affine), os.path.join(os.path.dirname(save_file) , 'gt_slice' + str(slice_range[0]) +'to' + str(slice_range[1])+'.nii.gz'))

#         # start to run
#         for k in range(0,run_iterations):
#             # find the correct slice number
#             start_slice = k * self.batch_size
#             if (k+1) * self.batch_size > self.generator.num_slices:
#                 end_slice = self.generator.num_slices
#             else:
#                 end_slice = (k+1) * self.batch_size
           
#             print('iteration ', k, 'correct slice num: ', start_slice, end_slice)
       
#             with torch.inference_mode():
#                 datas = next(self.cycle_dl) 
#                 data_input = datas[1]
#                 print('data input shape: ', data_input.shape)

#                 data_condition_save = torch.clone(data_input).numpy().squeeze()
#                 print(data_condition_save.shape)
#                 data_condition_save = np.rollaxis(data_condition_save, 0,3)
#                 print(data_condition_save.shape)
#                 data_condition_save = Data_processing.crop_or_pad(data_condition_save, [gt_img.shape[0], gt_img.shape[1], data_condition_save.shape[-1]], value = np.min(data_condition_save))
#                 data_condition_save = Data_processing.normalize_image(data_condition_save, normalize_factor = normalize_factor, image_max = maximum_cutoff, image_min = background_cutoff, invert = True)
#                 nb.save(nb.Nifti1Image(data_condition_save, gt.affine), os.path.join(os.path.dirname(save_file), 'condition.nii.gz'))

#                 data_input = data_input.to(torch.float32).to(device)
#                 pred_img = self.ema.ema_model(data_input)

#                 pred_img_residual = pred_img.detach().cpu().numpy().squeeze()
#                 pred_img_final = pred_img_residual + data_input.detach().cpu().numpy().squeeze()
                         
            
#             if pred_img_residual.ndim == 3:
#                 pred_img_residual = np.rollaxis(pred_img_residual, 0,3)
#                 pred_img_final = np.rollaxis(pred_img_final, 0,3)
#             if pred_img_residual.ndim == 2:
#                 pred_img_residual = np.expand_dims(pred_img_residual, axis = 2)
#                 pred_img_final = np.expand_dims(pred_img_final, axis = 2)

#             pred_img_residual = Data_processing.normalize_image(pred_img_residual, normalize_factor = normalize_factor, image_max = maximum_cutoff, image_min = background_cutoff, invert = True)
#             pred_img_final = Data_processing.normalize_image(pred_img_final, normalize_factor = normalize_factor, image_max = maximum_cutoff, image_min = background_cutoff, invert = True)
            
#             pred_img_residual_I[:,:, start_slice : end_slice] = pred_img_residual
#             pred_img_final_I[:,:, start_slice : end_slice] = pred_img_final

#         pred_img_residual_I = Data_processing.crop_or_pad(pred_img_residual_I, [gt_img.shape[0], gt_img.shape[1], pred_img_residual_I.shape[-1]], value = np.min(gt_img))
#         pred_img_residual_I = Data_processing.correct_shift_caused_in_pad_crop_loop(pred_img_residual_I)

#         pred_img_final_I = Data_processing.crop_or_pad(pred_img_final_I, [gt_img.shape[0], gt_img.shape[1], pred_img_final_I.shape[-1]], value = np.min(gt_img))
#         pred_img_final_I = Data_processing.correct_shift_caused_in_pad_crop_loop(pred_img_final_I)

#         # save
#         nb.save(nb.Nifti1Image(pred_img_final_I, gt.affine), save_file)
#         nb.save(nb.Nifti1Image(pred_img_residual_I, gt.affine), os.path.join(os.path.dirname(save_file),'pred_residual_slice' + str(slice_range[0]) +'to' + str(slice_range[1])+'.nii.gz'))
        
#         # also save gt and motion for comparision
#         if save_gt_motion:
#             motion_img = nb.load(motion_image_file).get_fdata()[:,:,slice_range[0]: slice_range[1]]
#             motion_img = Data_processing.cutoff_intensity(motion_img, cutoff_low = background_cutoff, cutoff_high = maximum_cutoff)
#             motion_img_save = nb.Nifti1Image(motion_img, gt.affine)
#             nb.save(motion_img_save, os.path.join(os.path.dirname(save_file), 'motion_slice' + str(slice_range[0]) +'to' + str(slice_range[1])+'.nii.gz'))

#             nb.save(nb.Nifti1Image(gt_img - motion_img, gt.affine), os.path.join(os.path.dirname(save_file) , 'gt_residual_slice' + str(slice_range[0]) +'to' + str(slice_range[1])+'.nii.gz'))

   
