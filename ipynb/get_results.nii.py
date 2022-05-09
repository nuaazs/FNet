from antspynet.utilities import lung_extraction
from monai.networks.blocks import Warp
import torch
import ants
import os
# from get_lung_volume import get_lung_mask
import numpy as np
import shutil

def plot_ct(img,slices,overlay=None,title=None):
    if overlay!=None:
        img_over = overlay - overlay.min()
    else:
        img_over = None
    img2 = img-img.min()
    img2.plot(axis=1,slices=slices,title=title,overlay=img_over,overlay_alpha=0.5)

def compute_volume(img):
    """
    Computes the area (number of pixels) of a binary mask and multiplies the pixels
    with the pixel dimension of the acquired CT image
    Args:
        lung_mask: binary lung mask
        pixdim: list or tuple with two values
    Returns: the lung area in mm^2
    """
    pixdim = img.spacing
    img = img.numpy()
    # img[img >= 0.5] = 1
    # img[img < 0.5] = 0
    lung_pixels = np.sum(img)
    return lung_pixels * pixdim[0] * pixdim[1]* pixdim[2]

def get_lung_mask(img):
    lung_result = lung_extraction(img, modality='ct', verbose=False)
    left_lung = lung_result['probability_images'][1]
    right_lung = lung_result['probability_images'][2]

    left_lung_volume = compute_volume(left_lung)
    right_lung_volume = compute_volume(right_lung)

    return {"left_lung_volume":left_lung_volume,
            "right_lung_volume":right_lung_volume,
            "left_lung":left_lung,
            "right_lung":right_lung}

def resample_dff(ddf_npy_filepath_1,ddf_npy_filepath_2,moving_filepath,fixed_filepath,warp,savepath=None,plot=False):
    moving_img = ants.image_read(moving_filepath)
    fixed_img = ants.image_read(fixed_filepath)



    ddf_1 = []
    for index in range(3):
        ddf_npy_1 = np.load(ddf_npy_filepath_1)[0,index]
        ddf_raw_1 = ants.from_numpy(ddf_npy_1)
        ddf_raw_1 = ants.resample_image(ddf_raw_1,moving_img.shape,True,4)
        ddf_1.append(ddf_raw_1.numpy())
    ddf_1 = torch.tensor(np.array(ddf_1))

    print(f"DDF 1 Shape:{ddf_1.shape}")
    
  

    ddf_2 = []
    for index in range(3):
        ddf_npy_2 = np.load(ddf_npy_filepath_2)[0,index]
        ddf_raw_2 = ants.from_numpy(ddf_npy_2)
        ddf_raw_2 = ants.resample_image(ddf_raw_2,moving_img.shape,True,4)
        ddf_2.append(ddf_raw_2.numpy())
    ddf_2 = torch.tensor(np.array(ddf_2))

    print(f"DDF 2 Shape:{ddf_2.shape}")


    img = torch.tensor(moving_img.numpy())
    img = torch.unsqueeze(img, dim=0)
    img = torch.unsqueeze(img, dim=0)
    img = (img+1000.)/3000.
    img[img<0]=0
    img[img>1]=1
    
    ddf_1 = torch.unsqueeze(ddf_1, dim=0)
    ddf_2 = torch.unsqueeze(ddf_2, dim=0)

    output_img_1 = warp(img,ddf_1).numpy()
    output_img_2 = warp(img,ddf_2).numpy()

    print(f"\tIMG shape:{img.shape}")
    print(f"\tDDF 1 shape:{ddf_1.shape}")
    print(f"\tOUT 1 shape:{output_img_1.shape}")
    print(f"\tDDF 2 shape:{ddf_2.shape}")
    print(f"\tOUT 2 shape:{output_img_2.shape}")
    output_img_1 = ants.from_numpy(output_img_1[0,0,:,:,:])
    output_img_1.set_spacing(moving_img.spacing)
    output_img_2 = ants.from_numpy(output_img_2[0,0,:,:,:])
    output_img_2.set_spacing(moving_img.spacing)

    slices = np.linspace((moving_img.shape[1])/4,(moving_img.shape[1])*3/5.,9,dtype=int)

    if plot:
        plot_ct(output_img_1,slices,None,"Fake CT 1")
        plot_ct(output_img_2,slices,None,"Real CT Transform")
        plot_ct(fixed_img,slices,None,"Real CT")
        plot_ct(moving_img,slices,None,"Input CT")

    
    if savepath:
        input_filename = moving_filepath.split("/")[-1]
        real_filename = fixed_filepath.split("/")[-1]
        t_index = ddf_npy_filepath_1.split("/")[-1].split(".")[0].split("_")[2]
        output_filename = ddf_npy_filepath_1.split("/")[-1].split(".")[0].split("_")[1]+"_"+t_index+".nii"
        os.makedirs(os.path.join(savepath,"fake"),exist_ok=True)
        os.makedirs(os.path.join(savepath,"real"),exist_ok=True)
        os.makedirs(os.path.join(savepath,"input"),exist_ok=True)
        os.makedirs(os.path.join(savepath,"real_transform"),exist_ok=True)
        ants.image_write(output_img_1,os.path.join(savepath,"fake",output_filename))
        ants.image_write(fixed_img,os.path.join(savepath,"real",real_filename))
        ants.image_write(output_img_2,os.path.join(savepath,"real_transform",real_filename))
        ants.image_write(moving_img,os.path.join(savepath,"input",input_filename))
   

    return ddf,moving_img,fixed_img,output_img_1,output_img_2


for filepath in sorted([os.path.join("/zhaosheng_data/4dct_test_transform/",_file) for _file in os.listdir("/zhaosheng_data/4dct_test_transform/") if ".nii" in _file]):#["/zhaosheng_data/4dct_test_transform/huyannian_t1.nii.gz"]:
    # "/zhaosheng_data/4dct_test_transform/huyannian_t1.nii.gz"
    
    filename = filepath.split("/")[-1].replace(".nii.gz","")
    # print(filepath)
    print(filename)
    try:
        ddf_test  = ants.image_read(filepath)
        npy = ddf_test.numpy().transpose(3,0,1,2)
        output_npy = np.expand_dims(npy,0)
        np.save(f"./4dct_test_transform/{filename}.npy",output_npy)
    except Exception as e:
        print(f"Error : {filename}")
        print(e)