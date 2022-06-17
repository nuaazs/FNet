
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


import ants
import os
import numpy as np
from utils.ddf_tools import warp_img_npy,warp_img_nii
from utils.lung import get_lung_mask
from IPython import embed
from utils.plot import plot_ct
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from utils.log import Logger
check_log = Logger('check_results.log')

def get_result(ddf_real,ddf_fake,moving,fixed,t_index,savepath=None,plot=False,get_lung=True):
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
    slices = np.array([50,55,60])*2
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
    SAVE_PNG_PATH = "./pngs_256"
    SAVE_FAKE_RESAMPLED_PATH = "/pan/4dct_256_resampled_fake"
    SAVE_LUNG_MASK_PATH = "/pan/4dct_256_lung_mask"
    os.makedirs(SAVE_LUNG_MASK_PATH,exist_ok=True)
    os.makedirs(SAVE_PNG_PATH,exist_ok=True)
    files = sorted([os.path.join(SAVE_FAKE_RESAMPLED_PATH,file) for file in os.listdir(SAVE_FAKE_RESAMPLED_PATH) if ".nii.gz" in file])
    pnames = sorted(list(set([file.split("/")[-1].split("_")[0] for file in files])))
    check_log.info(f'#Pnames:{pnames}')
    p_index = 1
    left_lung_fake_all,right_lung_fake_all = [],[]
    left_lung_real_all,right_lung_real_all = [],[]

    for pname in tqdm(pnames):
        check_log.info(f"#Start: No.{p_index} {pname}")
        left_lung_real = [pname]
        left_lung_fake = [pname]
        right_lung_real = [pname]
        right_lung_fake = [pname]
        # [pname,left_lung_0,left_lung_1, ... ,left_lung_9] -> left_lung
        # [pname,right_lung_0,right_lung_1, ... ,right_lung_9] -> right_lung
        for index in range(1,10):
            real_img_path = f"{SAVE_FAKE_RESAMPLED_PATH}/{pname}_t{index}_fake_fake.nii.gz"
            fake_img_path = f"{SAVE_FAKE_RESAMPLED_PATH}/{pname}_t{index}_fake_real.nii.gz"
            try:
                result = get_result_from_img(real_img_path=real_img_path,
                                            fake_img_path=fake_img_path,t_index=index)
            except Exception as e:
                check_log.info(f"Error!!! {pname} t{index}***********\n{e}\n***********")
                continue
            left_lung_real.append(result["lung_real"]["left_lung_volume"])
            left_lung_fake.append(result["lung_fake"]["left_lung_volume"])
            right_lung_real.append(result["lung_real"]["right_lung_volume"])
            right_lung_fake.append(result["lung_fake"]["right_lung_volume"])
            
            # !TODO save lung mask
            ants.image_write(result["lung_real"]["left_lung"],f"{SAVE_LUNG_MASK_PATH}/{pname}_t{index}_real_lung_left.nii.gz")
            ants.image_write(result["lung_real"]["right_lung"],f"{SAVE_LUNG_MASK_PATH}/{pname}_t{index}_real_lung_right.nii.gz")
            ants.image_write(result["lung_fake"]["left_lung"],f"{SAVE_LUNG_MASK_PATH}/{pname}_t{index}_fake_lung_left.nii.gz")
            ants.image_write(result["lung_fake"]["right_lung"],f"{SAVE_LUNG_MASK_PATH}/{pname}_t{index}_fake_lung_right.nii.gz")
            p_index += 1
        os.makedirs(os.path.join(SAVE_PNG_PATH,f"{pname}"),exist_ok=True)
        
        # embed()
        plt.figure(dpi=200)
        plt.title(f"Patient {pname} Left Lung")
        plt.plot(left_lung_real[1:],label="left_lung_real")
        plt.plot(left_lung_fake[1:],label="left_lung_fake")
        plt.legend()
        plt.savefig(os.path.join(SAVE_PNG_PATH,pname,f"{pname}_left_lung.png"))
        # plt.show()

        plt.figure(dpi=200)
        plt.title(f"Patient {pname} Right Lung")
        plt.plot(right_lung_real[1:],label="right_lung_real")
        plt.plot(right_lung_fake[1:],label="right_lung_fake")
        plt.legend()
        plt.savefig(os.path.join(SAVE_PNG_PATH,pname,f"{pname}_right_lung.png"))
        # plt.show()
        
        left_lung_real_all.append(left_lung_real)
        check_log.info(f"\t***\tLeft Lung Real: {left_lung_real}")
        left_lung_fake_all.append(left_lung_fake)
        check_log.info(f"\t***\tLeft Lung Fake: {left_lung_fake}")
        right_lung_real_all.append(right_lung_real)
        check_log.info(f"\t***\tRight Lung Real: {right_lung_real}")
        right_lung_fake_all.append(right_lung_fake)
        check_log.info(f"\t***\tRight Lung Fake: {right_lung_fake}")
        
    np.save("right_lung_fake_all_256.npy",right_lung_fake_all)
    np.save("left_lung_fake_all_256.npy",left_lung_fake_all)
    np.save("right_lung_real_all_256.npy",right_lung_real_all)
    np.save("left_lung_real_all_256.npy",left_lung_real_all)
