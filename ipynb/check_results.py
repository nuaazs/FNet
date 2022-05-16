
# ██╗██╗███╗   ██╗████████╗
# ██║██║████╗  ██║╚══██╔══╝
# ██║██║██╔██╗ ██║   ██║
# ██║██║██║╚██╗██║   ██║
# ██║██║██║ ╚████║   ██║
# ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝
# @Time    : 2022-05-12 20:44:37.000-05:00
# @Author  : 𝕫𝕙𝕒𝕠𝕤𝕙𝕖𝕟𝕘
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : http://iint.icu/
# @File    : /home/zhaosheng/paper4/ipynb/check_results.py
# @Describe: check 4d ct results

import torch
import ants
import os
import numpy as np
from utils.ddf_tools import warp_img_npy,warp_img_nii
from utils.lung import get_lung_mask
from utils.plot import plot_ct
import os
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename='./check_results.log', format="%(levelname)s %(asctime)s - %(message)s", level=logging.INFO)

def get_result(ddf_real,ddf_fake,moving,fixed,t_index,savepath=None,plot=True,get_lung=True):
    result_real = warp_img_nii(fixed_filepath=fixed,moving_filepath=moving,ddf_nii_filepath=ddf_real)
    result_fake = warp_img_npy(fixed_filepath=fixed,moving_filepath=moving,ddf_npy_filepath=ddf_fake,real_ddf_filepath=ddf_real)

    result={
        "fixed":result_real["fixed"],
        "moved_real":result_real["moved"],
        "moved_fake":result_fake["moved"],
        "moving":result_real["moving"],
        "lung_real":{},
        "lung_fake":{},
        "lung_moving":{}
    }

    slices = [50,55,60,64,70,75,80]
    if plot:
        plot_ct(result_real["fixed"],slices,None,f"{pname} T{t_index} Fixed")
        plot_ct(result_real["moving"],slices,None,f"{pname} T{t_index} moving")
        plot_ct(result_real["moved"],slices,None,f"{pname} T{t_index} moved real")
        plot_ct(result_fake["moved"],slices,None,f"{pname} T{t_index} moved fake")
    if get_lung:
        result["lung_real"] = get_lung_mask(result_real["moved"])
        result["lung_fake"] = get_lung_mask(result_fake["moved"])
        result["lung_moving"] = get_lung_mask(result_real["moving"])
    if savepath:
        os.makedirs(f"{savepath}/{pname}/{t_index}", exist_ok=True)
        ants.image_write(result_real["fixed"],filename=f"{savepath}/{pname}/{t_index}/fixed.nii.gz")
        ants.image_write(result_real["moving"],filename=f"{savepath}/{pname}/{t_index}/moving.nii.gz")
        ants.image_write(result_real["moved"],filename=f"{savepath}/{pname}/{t_index}/moved_real.nii.gz")
        ants.image_write(result_fake["moved"],filename=f"{savepath}/{pname}/{t_index}/moved_fake.nii.gz")
    return result


def get_result_from_img(real_img_path,fake_img_path,t_index,plot=True,get_lung=True):
    real = ants.image_read(real_img_path)
    fake = ants.image_read(fake_img_path)
    result = {}
    slices = [50,55,60,64,70,75,80]
    if plot:
        plot_ct(real+1000,slices,None,f"{pname} T{t_index} moved real")
        plot_ct(fake+1000,slices,None,f"{pname} T{t_index} moved fake")
    if get_lung:
        result["lung_real"] = get_lung_mask(real)
        result["lung_fake"] = get_lung_mask(fake)
    return result

def sortddf(item):
    return item.split("/")[-1].split(".")[0].split("_ddf")[-1]

if __name__ == "__main__":
    # ddf_path = "/home/zhaosheng/paper4/outputs/A2B_0515/34/ddfs"
    lung_png_path = "./pngs"
    img_save_path = "./imgs"

    SAVE_FAKE_RESAMPLED_PATH = "/media/wurenyao/TOSHIBA EXT/4dct_512_resampled_fake"

    files = sorted([os.path.join(SAVE_FAKE_RESAMPLED_PATH,file) for file in os.listdir(SAVE_FAKE_RESAMPLED_PATH) if ".nii.gz" in file])
    pnames = set([file.split("/")[-1].split("_")[0] for file in files])
    
    print(pnames)
    p_index = 1
    left_lung_fake_all,right_lung_fake_all = [],[]
    left_lung_real_all,right_lung_real_all = [],[]

    for pname in tqdm(pnames):
        logging.info(f"#Start: No.{p_index} {pname}")
        left_lung_real = [pname]
        left_lung_fake = [pname]
        right_lung_real = [pname]
        right_lung_fake = [pname]
        

        # [pname,left_lung_0,left_lung_1, ... ,left_lung_9] -> left_lung
        # [pname,right_lung_0,right_lung_1, ... ,right_lung_9] -> right_lung
        for index in range(1,10):
            # if index<=5:
            #     ddf_path = "/home/zhaosheng/paper4/outputs/A2B_0515/34/ddfs"
            # else:
            #     ddf_path = "/home/zhaosheng/paper4/outputs/B2A_0515/34/ddfs"

            # fixed = f"/dataset1/4dct_0510/resampled/{pname}_t{index}_resampled.nii"
            # moving = f"/dataset1/4dct_0510/resampled/{pname}_t0_resampled.nii"
            # real_ddf = os.path.join("/dataset1/4dct_0510/transform",f"{pname}_t{index}_Warp.nii.gz")
            # fake_ddf = os.path.join(ddf_path,f"34_{pname}_ddf{index}.npy")
            # logging.info(f"\t\tT{index} fixed : {fixed}")
            # logging.info(f"\t\tT{index} moving: {moving}")
            # logging.info(f"\t\tT{index} ddf_r : {real_ddf}")
            # logging.info(f"\t\tT{index} ddf_f : {fake_ddf}")
            
            # result = get_result(
            #     ddf_real=real_ddf,
            #     ddf_fake=fake_ddf,
            #     moving=moving,
            #     fixed=fixed,
            #     t_index=index,
            #     savepath=img_save_path,
            #     plot=False,
            # )
            real_img_path = f"{SAVE_FAKE_RESAMPLED_PATH}/{pname}_t{index}_fake_fake.nii.gz"
            fake_img_path = f"{SAVE_FAKE_RESAMPLED_PATH}/{pname}_t{index}_fake_real.nii.gz"

            result = get_result_from_img(real_img_path=real_img_path,
                                        fake_img_path=fake_img_path,t_index=index)

            left_lung_real.append(result["lung_real"]["left_lung_volume"])
            left_lung_fake.append(result["lung_fake"]["left_lung_volume"])
            right_lung_real.append(result["lung_real"]["right_lung_volume"])
            right_lung_fake.append(result["lung_fake"]["right_lung_volume"])
            p_index += 1
        os.makedirs(os.path.join(lung_png_path,f"{pname}"),exist_ok=True)
        plt.figure(dpi=200)
        plt.title(f"Patient {pname} Left Lung")
        plt.plot(left_lung_real[1:],label="left_lung_real")
        plt.plot(left_lung_fake[1:],label="left_lung_fake")
        plt.legend()
        plt.savefig(os.path.join(lung_png_path,f"{pname}_left_lung.png"))
        # plt.show()

        plt.figure(dpi=200)
        plt.title(f"Patient {pname} Right Lung")
        plt.plot(right_lung_real[1:],label="right_lung_real")
        plt.plot(right_lung_fake[1:],label="right_lung_fake")
        plt.legend()
        plt.savefig(os.path.join(lung_png_path,f"{pname}_right_lung.png"))
        # plt.show()
        
        left_lung_real_all.append(left_lung_real)
        logging.info(f"\tLeft Lung Real: {left_lung_real}")
        left_lung_fake_all.append(left_lung_fake)
        logging.info(f"\t\tLeft Lung Fake: {left_lung_fake}")
        right_lung_real_all.append(right_lung_real)
        logging.info(f"\t\tRight Lung Real: {right_lung_real}")
        right_lung_fake_all.append(right_lung_fake)
        logging.info(f"\t\tRight Lung Fake: {right_lung_fake}")

        break
