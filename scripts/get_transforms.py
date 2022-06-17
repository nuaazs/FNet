# ██╗██╗███╗   ██╗████████╗
# ██║██║████╗  ██║╚══██╔══╝
# ██║██║██╔██╗ ██║   ██║
# ██║██║██║╚██╗██║   ██║
# ██║██║██║ ╚████║   ██║
# ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝
# @Time    : 2022-05-10 11:34:15.000-05:00
# @Author  : 𝕫𝕙𝕒𝕠𝕤𝕙𝕖𝕟𝕘
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : http://iint.icu/
# @File    : /dataset1/4dct/get_transforms.py
# @Describe: Get transforms and reshape nii files.

from asyncio.log import logger
from click import style
from cv2 import log
from datasets_maker import DatasetsMaker as DM
import os
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import os
import re
import pydicom
import shutil
import numpy as np
from PIL import Image
import SimpleITK as sitk
import nibabel as nib
import ants
from time import *
# ANTs param
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "80"
os.environ["ANTS_RANDOM_SEED"] = "3"

import logging
logging.basicConfig(filename="get_transform.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("** Start **")

# DatasetsMaker

def get_transform(moving_file, fixed_file,type_of_transform,plots_path,transform_path,resampled_path,fixed_shape=(128,128,64)):
    """get transform between two antsImages and save reshaped image."

    Args:
        moving_file (antsImage): moving Image t0
        fixed_file (antsImage): fixed Image tn
        type_of_transform (str): type_of_tr
        plots_path (str): where to save the pngs
        transform_path (str): where to save the transforms
        resampled_path (str): where to save the resampled images
        fixed_shape (tuple, optional): image fixed_shape. Defaults to (128,128,64).

    Returns:
        None
    """
    begin_time = time()
    filename = fixed_file.split("/")[-1].split(".")[-2]
    print(f"\n=> Now loading:{filename}")
    moving = ants.image_read(moving_file)
    fixed = ants.image_read(fixed_file)

    # Resample image        
    if fixed_shape:
        moving_resampled = ants.resample_image(moving,fixed_shape,True,4)
        fixed_resampled = ants.resample_image(fixed,fixed_shape,True,4)
    else:
        new_spacing = np.array(moving.spacing)*4
        moving_resampled = ants.resample_image(moving,new_spacing,False,4)
        fixed_resampled = ants.resample_image(fixed,new_spacing,False,4)

    
    if resampled_path:
        os.makedirs(resampled_path, exist_ok=True)
        ants.image_write(moving_resampled, os.path.join(resampled_path, filename.split("_")[0]+"_t0_resampled.nii"))
        ants.image_write(fixed_resampled, os.path.join(resampled_path, filename+"_resampled.nii"))

    

    print(f"\t-> Registration ...")
    reg = ants.registration(
        fixed=fixed_resampled, moving=moving_resampled, type_of_transform=type_of_transform)
    moved = reg["warpedmovout"]
    
    # print(f"\t-> Plot ...")
    # if plots_path:
    #     os.makedirs(plots_path,exist_ok=True)
    #     moved_plot = moved + 1000
    #     fixed_resampled_plot = fixed_resampled + 1000
    #     moved_plot.plot(title='moved', axis=1, cbar=True,
    #         filename=os.path.join(plots_path, filename+"_moved.png"))
    #     ants.plot(moved_plot, overlay=fixed_resampled_plot, overlay_cmap='hot', overlay_alpha=0.5,
    #             axis=1, cbar=True, filename=os.path.join(plots_path, filename+"_overlay_1.png"))
    #     ants.plot(moved_plot, overlay=fixed_resampled_plot, overlay_cmap='hot', overlay_alpha=0.5,
    #             axis=0, cbar=True, filename=os.path.join(plots_path, filename+"_overlay_0.png"))

    
    
    print(f"\t-> Save transforms ...")
    if transform_path:
        os.makedirs(transform_path, exist_ok=True)
        for output_file in reg["fwdtransforms"]:
            if "nii" in output_file:
                shutil.move(output_file, f"{transform_path}/{filename}_Warp.nii.gz")
            if "mat" in output_file:
                shutil.move(output_file, f"{transform_path}/{filename}_GenericAffine.mat")
    end_time = time()
    run_time = end_time-begin_time
    print(f"\t-> Used:{run_time}s")
    

def getTransform(item):
    patient,i = item
    # get_transform(moving_file       = f"/media/wurenyao/TOSHIBA EXT/4dct_niis/t0/{patient}_t0.nii",
    #                  fixed_file        = f"/media/wurenyao/TOSHIBA EXT/4dct_niis/t{i}/{patient}_t{i}.nii",
    #                  type_of_transform = "SyNAggro",
    #                  plots_path        = "/dataset1/4dct_0510/4dct_reg_plots/",
    #                  transform_path    = "/dataset1/4dct_0510/transform/",
    #                  resampled_path    = "/dataset1/4dct_0510/resampled/",
    #                  fixed_shape       = (128,128,64)
    #                 )
    get_transform(moving_file       = f"/home/zhaosheng/4dct_test_nii/{patient}/{patient}_t0.nii",
                     fixed_file        = f"/home/zhaosheng/4dct_test_nii/{patient}/{patient}_t{i}.nii",
                     type_of_transform = "SyNAggro",
                     plots_path        = "/dataset1/4dct_0510_test/4dct_reg_plots/",
                     transform_path    = "/dataset1/4dct_0510_test/transform/",
                     resampled_path    = "/dataset1/4dct_0510_test/resampled/",
                     fixed_shape       = (128,128,64)
                    )

def make_data_multi(items):
    print(items[0])
    pool = ThreadPool()
    pool.map(getTransform, items)
    pool.close()
    pool.join()

def make_data(items):
    for item in tqdm(items):
        try:
            getTransform(item)
        except Exception as e:
            print(f"Pass : {item}\n=====================\n{e}\n=====================")

if __name__ == "__main__":
    files = []
    # raw nii path (512*512*x)
    # files = os.listdir("/media/wurenyao/TOSHIBA EXT/4dct_niis/t9")
    pnames = os.listdir("/home/zhaosheng/4dct_test_nii/")
    patients_dict = {}
    for pname in pnames:
        files+=os.listdir(f"/home/zhaosheng/4dct_test_nii/{pname}")
    for _file in files:
        patient = _file.split("_")[0]
        if patient in patients_dict:
            patients_dict[patient].append(_file.split("_")[-1].split(".")[0])
        else:
            patients_dict[patient] = [_file.split("_")[-1].split(".")[0]]
    print(f"Patients: {patients_dict}")

    items = []

    for patient in tqdm(patients_dict.keys()):
        try:
            for i in range(1,10):
                items.append([patient,i])
        except Exception as e:
            print(e)
    
    #make_data_multi(items)
    make_data(items)