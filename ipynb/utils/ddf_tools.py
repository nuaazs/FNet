import torch
import numpy as np
import ants


def load_npy_ddf(npy_array_filepath,shape,spacing,savepath="./temp.nii.gz"):
    npy_array = np.load(npy_array_filepath,allow_pickle=True)[0]
    if npy_array.shape[0] != shape:
        resample_3d_npy(npy_array,shape)
    ddf_img = ants.from_numpy(npy_array)
    ants.set_spacing(ddf_img,(1,spacing[0],spacing[1],spacing[2]))
    ants.image_write(ddf_img,savepath)
    return ddf_img,savepath

def warp_img_npy(fixed_filepath,moving_filepath,ddf_npy_filepath):
    fixed = ants.image_read(fixed_filepath)
    moving = ants.image_read(moving_filepath)
    ddf_img,savepath = load_npy_ddf(ddf_npy_filepath,fixed.shape,fixed.spacing)
    moved_img = ants.apply_transforms( fixed=fixed, moving=moving,transformlist=[savepath])
    result = {
        "fixed":fixed,
        "moving":moving,
        "ddf":ddf_img,
        "moved":moved_img
    }
    return result

def warp_img_nii(fixed_filepath,moving_filepath,ddf_nii_filepath):
    fixed = ants.image_read(fixed_filepath)
    moving = ants.image_read(moving_filepath)
    ddf_img = ants.image_read(ddf_nii_filepath)
    if ddf_img.shape != moving.shape:
        ddf_npy = ddf_img.numpy().transpose(3,0,1,2)
        ddf_npy_reshaped = resample_3d_npy(ddf_npy,moving.shape)
        ddf_img_reshaped = ants.from_numpy(ddf_npy_reshaped)
        ants.set_spacing(ddf_img_reshaped,(1,moving.spacing[0],moving.spacing[1],moving.spacing[2]))
        ants.image_write(ddf_img_reshaped,"./temp_real_ddf.nii")
    moved_img = ants.apply_transforms( fixed=fixed, moving=moving,transformlist=["./temp_real_ddf.nii"])
    result = {
        "fixed":fixed,
        "moving":moving,
        "ddf":ddf_img,
        "moved":moved_img
    }
    return result

def resample_3d_npy(npy_array,shape):
    ddf_1 = []
    for index in range(3):
        ddf_npy_1 = npy_array[0,index]
        ddf_raw_1 = ants.from_numpy(ddf_npy_1)
        ddf_raw_1 = ants.resample_image(ddf_raw_1,shape,True,4)
        ddf_1.append(ddf_raw_1.numpy())
    ddf_1 = np.array(ddf_1)
    return ddf_1
