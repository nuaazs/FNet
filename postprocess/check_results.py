import ants
import torch
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from utils.dsb_lung import get_lung_mask
from scipy.signal import savgol_filter
from IPython import embed

from cfg import real_nii_path
from cfg import tumor_path
from cfg import output_path
from cfg import EPOCH


def get_result(output_path, epoch, type="_ddf_fake"):
    return sorted(
        [
            os.path.join(output_path, _file)
            for _file in os.listdir(output_path)
            if _file.startswith(f"{epoch}_") and str(type) in _file
        ],
        key=lambda x: x.split("_")[1],
    )


def norm_list(data, input_volume):
    data = np.array(data) + input_volume
    result = (
        (np.array(data) - np.array(data).min())
        * 100
        / (np.array(data).max() - np.array(data).min())
    )
    result[0] = 100
    return result


def plot_img(img, add=True):
    result = get_lung_mask(img)
    if add:
        img = img + 1000
    ants.plot(img, axis=1, nslices=3, slices=[40, 50, 55], cbar="gray")
    return result["lung_volume"]


def get_lung_volume(img):
    result = get_lung_mask(img)
    return result["lung_volume"]


if __name__ == "__main__":
    real_result_all = []
    fake_result_all = []

    fake_images = get_result(output_path, EPOCH, "_image_fake")
    real_images = get_result(output_path, EPOCH, "_image_real")
    fake_ddfs = get_result(output_path, EPOCH, "_ddf_fake")
    real_ddfs = get_result(output_path, EPOCH, "_ddf_real")

    for index in range(len(real_ddfs)):
        pname = fake_images[index].split("/")[-1].split("_")[1]
        fake_images_npy = np.load(fake_images[index])[0]
        real_images_npy = np.load(real_images[index])[0]
        fake_ddfs_npy = np.load(fake_ddfs[index])[0].reshape(9, 3, 128, 128, 64)
        real_ddfs_npy = np.load(real_ddfs[index])[0].reshape(9, 3, 128, 128, 64)
        # print(f"ddf shape : {real_ddfs_npy.shape}")
        assert real_ddfs_npy.shape == (9, 3, 128, 128, 64), "real ddf shape error!"
        assert fake_ddfs_npy.shape == (9, 3, 128, 128, 64), "fake ddf shape error!"
        input_image = ants.image_read(
            os.path.join(real_nii_path, f"{pname}_t0_resampled.nii")
        )
        input_v = get_lung_volume(input_image)
        fake_v_list = [input_v]
        real_v_list = [input_v]

        for i in range(9):
            fake_image = fake_images_npy[i] * 5000 - 500
            real_image = real_images_npy[i] * 5000 - 500

            fake_ddf = fake_ddfs_npy[i]
            real_ddf = real_ddfs_npy[i]
            real_image_nii = ants.image_read(
                os.path.join(real_nii_path, f"{pname}_t{i}_resampled.nii")
            )
            fake_image = real_image_nii.new_image_like(fake_image)
            real_image = real_image_nii.new_image_like(real_image)
            os.makedirs(f"./result_nii/real/{EPOCH}/", exist_ok=True)
            os.makedirs(f"./result_nii/fake/{EPOCH}/", exist_ok=True)
            ants.image_write(real_image, f"./result_nii/real/{EPOCH}/{pname}_t{i}.nii")
            ants.image_write(fake_image, f"./result_nii/fake/{EPOCH}/{pname}_t{i}.nii")
            fake_v = get_lung_volume(fake_image)
            real_v = get_lung_volume(real_image)
            fake_v_list.append(fake_v)
            real_v_list.append(real_v)
        print(fake_v_list)
        print(real_v_list)

        # fake_v_list = savgol_filter(fake_v_list, 3, 2, mode= 'nearest')
        # real_v_list = savgol_filter(real_v_list, 3, 2, mode= 'nearest')
        # fake_v_list = norm_list(fake_v_list,input_volume=input_v)
        # real_v_list = norm_list(real_v_list,input_volume=input_v)
        fake_v_list_2 = savgol_filter(fake_v_list, 5, 3, mode="nearest")
        real_v_list_2 = savgol_filter(real_v_list, 5, 3, mode="nearest")
        real_result_all.append(real_v_list_2)
        fake_result_all.append(fake_v_list_2)

    fake_result_all = np.array(fake_result_all)
    real_result_all = np.array(real_result_all)
    np.save(f"real_{EPOCH}", real_result_all)
    np.save(f"fake_{EPOCH}", fake_result_all)
