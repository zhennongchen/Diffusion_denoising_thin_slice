#!/usr/bin/env python

import importlib
import CTProjector.src.ct_projector
importlib.reload(CTProjector.src.ct_projector)

import CTProjector.src.ct_projector.projector.cupy as ct_projector
import CTProjector.src.ct_projector.projector.cupy.fan_equiangular as ct_fan 
import CTProjector.src.ct_projector.projector.numpy as numpy_projector
import CTProjector.src.ct_projector.projector.numpy.fan_equiangluar as numpy_fan
import CTProjector.src.ct_projector.projector.cupy.parallel as ct_para
import CTProjector.src.ct_projector.projector.numpy.parallel as numpy_para

import numpy as np
import cupy as cp
import os
import HeadCT_motion_correction_PAR.functions_collection as ff
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform
import glob as gb
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


def define_forward_projector(img,spacing,total_view,du_nyquist = 1):
    projector = ct_projector.ct_projector()
    projector.from_file('./projector_fan.cfg')
    projector.nx = img.shape[3]
    projector.ny = img.shape[2]
    projector.nz = 1
    projector.nv = 1
    projector.dx = spacing[2]
    projector.dy = spacing[1]
    projector.dz = spacing[0]
    projector.nview = total_view
    if du_nyquist != 0:
        nyquist = projector.dx * projector.dsd / projector.dso / 2
        projector.du = nyquist * du_nyquist

    # for k in vars(projector):
    #     print (k, '=', getattr(projector, k))
    return projector


def backprojector(img,spacing, du_nyquist = 1):
    fbp_projector = numpy_projector.ct_projector()
    fbp_projector.from_file('./projector_fan.cfg')
    fbp_projector.nx = img.shape[3]
    fbp_projector.ny = img.shape[2]
    fbp_projector.nz = 1
    fbp_projector.nv = 1
    fbp_projector.dx = spacing[2]
    fbp_projector.dy = spacing[1]
    fbp_projector.dz = spacing[0]

    if du_nyquist != 0:
        nyquist = fbp_projector.dx * fbp_projector.dsd / fbp_projector.dso / 2
        fbp_projector.du = nyquist * du_nyquist

    return fbp_projector


def fp_static(img,angles,projector, geometry):
    # cp.cuda.Device(0).use()
    # ct_projector.set_device(0)
    origin_img = img[0, ...]
    origin_img = origin_img[:, np.newaxis, ...]
    cuimg = cp.array(origin_img, cp.float32, order = 'C')
    cuangles = cp.array(angles, cp.float32, order = 'C')

    if geometry[0:2] == 'fa':
        projector.set_projector(ct_fan.distance_driven_fp, angles=cuangles, branchless=False)
        numpy_projector.set_device(0)
    else:
        projector.set_projector(ct_para.distance_driven_fp, angles = cuangles,branchless = False)
        numpy_projector.set_device(0)

    # forward projection
    cufp = projector.fp(cuimg, angles = cuangles)
    fp = cufp.get()

    return fp



def filtered_backporjection(projection,angles,projector,fbp_projector, geometry, back_to_original_value = True):
    # z_axis = True when z_axis is the slice, otherwise x-axis is the slice

    cuangles = cp.array(angles, cp.float32, order = 'C')
    if geometry[0:2] == 'fa':
        fprj = numpy_fan.ramp_filter(fbp_projector, projection, filter_type='RL')
        projector.set_backprojector(ct_fan.distance_driven_bp, angles=cuangles, is_fbp=True)
    elif geometry[0:2] == 'pa':
        fprj = numpy_para.ramp_filter(fbp_projector, projection, filter_type='RL')
        projector.set_backprojector(ct_para.distance_driven_bp, angles=cuangles, is_fbp=True)
    else:
        ValueError('wrong geometry')

    cufprj = cp.array(fprj, cp.float32, order = 'C')
    curecon = projector.bp(cufprj)
    recon = curecon.get()
    recon = recon[:,0,...]

    if back_to_original_value == True:
        recon = recon / 0.019 * 1000 - 1024

    return recon

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
        print('doing poisson noise')

        return prj_noisy.astype(np.float32)

    return prj.astype(np.float32)