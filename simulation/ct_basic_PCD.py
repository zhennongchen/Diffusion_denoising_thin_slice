#!/usr/bin/env python

import ct_projector.projector.numpy.parallel as ct_para

import numpy as np
import cupy as cp
import copy
import nibabel as nb

# function: generate angle list
def get_angles_zc(nview, total_angle,start_angle):
    return np.arange(0, nview, dtype=np.float32) * (total_angle / 180 * np.pi) / nview + (start_angle / 180 * np.pi)


# for nibabel:
def basic_image_processing(filename , convert_value = True, header = False):
    ct = nb.load(filename)
    spacing = ct.header.get_zooms()
    img = ct.get_fdata()
    
    if convert_value == True:
        img = (img.astype(np.float32) + 1024) / 1000 * 0.019 
        img[img < 0] = 0
        
    img = np.rollaxis(img,-1,0)

    spacing = np.array(spacing[::-1])

    if header == False:
        return img,spacing,ct.affine
    else:
        return img, spacing, ct.affine, ct.header


def define_forward_projector_pcd(img,spacing, file_name = './pcd_parallel_6x5_512.cfg'):
    projector = ct_para.ct_projector()
    projector.from_file(file_name)
    projector.dx = spacing[1]
    projector.dy = spacing[2]
    projector.dz = spacing[0]
    projector.nx = img.shape[1]
    projector.ny = img.shape[2]
    projector.dv = projector.dz
    # print('[projector]')
    # for k in vars(projector):
    #     print(k, '=', getattr(projector, k), flush=True)
    return projector


def add_poisson_noise(prj: np.ndarray, N0: float, dose_factor: float):
    """
    Add realistic Poisson noise to the projection data.

    Args:
        prj (np.ndarray): Log-transformed projection data.
        N0 (float): Initial photon count (higher N0 means lower noise).
        dose_factor (float): Scaling factor for dose (0-1, where 1 is full dose).
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Noisy projection data.
    """

    if N0 > 0 and dose_factor < 1:
        # Convert log projection back to photon counts
        I_noisy = np.random.poisson(lam=(N0 * dose_factor * np.exp(-prj)))

        # Prevent log(0) issues by setting minimum value
        I_noisy = np.maximum(I_noisy, 1)

        # Convert back to log domain
        prj_noisy = -np.log(I_noisy / (N0 * dose_factor))

        return prj_noisy.astype(np.float32)

    return prj.astype(np.float32)

def add_gaussian_noise(prj: np.ndarray, N0: float, dose_factor: float):

    # add noise
    if N0 > 0 and dose_factor < 1:
        prj = prj + np.sqrt((1 - dose_factor) / dose_factor * np.exp(prj) / N0) * np.random.normal(size=prj.shape)
        prj = prj.astype(np.float32)

    return prj

def get_additional_filter_to_rl(filename, nu, du, nview, ninterp=20):
    '''
    Compose the filter that is on top of the RL filter
    '''
    custom_filter = np.fromfile(filename, np.float32)

    # these are the parameters after interleaving
    # the filter will be applied on the interleaved data
    nu *= 2
    du /= 2

    # compose rl filter
    rl_filter = np.zeros([2 * nu - 1], np.float32)
    k = np.arange(len(rl_filter)) - (nu - 1)
    for i in range(len(rl_filter)):
        if k[i] == 0:
            rl_filter[i] = 1 / (4 * du * du)
        elif k[i] % 2 != 0:
            rl_filter[i] = -1 / (np.pi * np.pi * k[i] * k[i] * du * du)
    frl_filter = np.fft.fft(rl_filter, len(custom_filter))
    frl_filter = np.abs(frl_filter)

    frl_filter = frl_filter * len(frl_filter) / nview * du * 2

    ratio = custom_filter / frl_filter

    # A smooth transition from 1 to the target ratio to avoid instability at the center
    ratio_fix = ratio.copy()
    ratio_fix[:ninterp] = 1 + (ratio[ninterp] - 1) * np.arange(ninterp) / ninterp
    ratio_fix = ratio_fix.astype(np.float32)
    ratio_fix[len(ratio_fix) // 2:] = ratio_fix[len(ratio_fix) // 2:0:-1]

    # plt.figure()
    # plt.plot(frl_filter)
    # plt.plot(custom_filter)
    # plt.plot(ratio)
    # plt.plot(ratio_fix)
    # plt.show()

    return ratio_fix

