# ██╗██╗███╗   ██╗████████╗
# ██║██║████╗  ██║╚══██╔══╝
# ██║██║██╔██╗ ██║   ██║
# ██║██║██║╚██╗██║   ██║
# ██║██║██║ ╚████║   ██║
# ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝
# @Time    : 2022-05-17  10:12:26
# @Author  : 𝕫𝕙𝕒𝕠𝕤𝕙𝕖𝕟𝕘
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : http://iint.icu/
# @File    : /home/zhaosheng/paper4/ipynb/utils/tumor.py
# @Describe: Get tumor locations.

import ants
import numpy as np
import torch
from utils.log import Logger
import matplotlib.pyplot as plt
from monai.networks.blocks import Warp

warp_layer = Warp()


def getLoc(_npy):
    _npy = np.array(_npy)
    loc = np.mean(np.argwhere(_npy == 1), axis=0)
    return loc


def get_tumor_location(nrrd_file_path, real_ddf_file_path, fake_ddf_file_path):
    """get tumor file and calc the tumor body centered

    Args:
        nrrd_file_path (string): nrrd file path for tumor.
        real_ddf_file_path (string): real ddf img nii file path
        fake_ddf_file_path (string): fake ddf img nii file path
    """
    # TODO:计算肿瘤坐标
    tumor_img = ants.image_read(nrrd_file_path)
    tumor_torch = (
        torch.tensor(tumor_img.numpy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
    )

    loc_before = getLoc(tumor_img.numpy())
    print(f"\t\t-> Tumor Location Before Warp {loc_before}")

    # Do Registration

    real_ddf_array = np.load(real_ddf_file_path, allow_pickle=True)
    fake_ddf_array = np.load(fake_ddf_file_path, allow_pickle=True)
    print(real_ddf_array.shape)
    print(fake_ddf_array.shape)
    tumor_warped_real = tumor_img.new_image_like(
        warp_layer(tumor_torch, torch.tensor(real_ddf_array).cuda())
        .cpu()
        .detach()
        .numpy()[0, 0]
    )
    tumor_warped_fake = tumor_img.new_image_like(
        warp_layer(tumor_torch, torch.tensor(fake_ddf_array).cuda())
        .cpu()
        .detach()
        .numpy()[0, 0]
    )

    tumor_warped_real = ants.apply_transforms(
        fixed=tumor_img, moving=tumor_img, transformlist=[real_ddf_file_path]
    )
    tumor_warped_fake = ants.apply_transforms(
        fixed=tumor_img, moving=tumor_img, transformlist=[fake_ddf_file_path]
    )
    loc_after_real = getLoc(tumor_warped_real.numpy())
    print(f"\t\t-> Tumor Locaation Before Warp {loc_after_real}")
    loc_after_fake = getLoc(tumor_warped_fake.numpy())
    print(f"\t\t-> Tumor Locaation Before Warp {loc_after_fake}")
    return {
        "loc_before": loc_before,
        "loc_after_real": loc_after_real,
        "loc_after_fake": loc_after_fake,
    }


def plot(array, ax, title, tumor=None):
    img = array.numpy()[0, 0, :, 307, ::-1].transpose(1, 0)

    if tumor != None:
        print(tumor.shape)
        assert tumor.shape == array.shape, "Tumor size error"
        # img[(np.array(tumor)[0,0,:,307,::-1]).transpose(1,0)==1] = 3000
        img = (np.array(tumor)[0, 0, :, 307, ::-1]).transpose(1, 0)

    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return 0


def plot_ddf(array, ax, title):
    ax.imshow(array)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return 0


def show_ddf(ddf):
    plt.figure(dpi=200)
    for index in range(3):
        plot_ddf(
            ddf.numpy()[0, index, :, 307, ::-1].transpose(1, 0),
            plt.subplot(1, 3, index + 1),
            f"Axis:{index}",
        )
    plt.show()


if __name__ == "__main__":
    pass
