# coding = utf-8
# @Time    : 2022-10-10  15:04:19
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 利用DDF，生成虚拟4DCT数据集，这样做的一个好处是DDF便可以成为真实DDF.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ants
import os
from IPython import embed
from sympy import arg
from monai.networks.blocks import Warp
import torch

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--ddf_dir", type=str, default="/mnt/zhaosheng/FNet/data/ddfs", help=""
)
parser.add_argument(
    "--img_dir", type=str, default="/mnt/zhaosheng/4dct/resampled", help=""
)
parser.add_argument(
    "--img_output_dir", type=str, default="/mnt/zhaosheng/4dct/output", help=""
)
args = parser.parse_args()

warp = Warp()

# STEP 1: 读取DDF
ddf_dir = args.ddf_dir
patients = sorted(
    [
        ddf.split("_")[1]
        for ddf in os.listdir(ddf_dir)
        if ".npy" in ddf and "49" in ddf and "_9_0" in ddf
    ]
)

for patient in patients:
    # STEP 2: 读取4DCT T0
    t0_img = ants.image_read(os.path.join(args.img_dir, f"{patient}_t0_resampled.nii"))
    t0 = torch.from_numpy(t0_img.numpy()).unsqueeze(0).unsqueeze(0)
    # STEP 3: 读取4DCT DDF 列表
    imgs = [t0.numpy()]
    for index in range(1, 10):
        print(f"Now processing {patient} T{index} ... ")
        # STEP 4: 读取DDF
        ddf = np.load(os.path.join(ddf_dir, f"49_{patient}_{index}_0.npy"))

        print(f"Image shape: {t0.shape}, DDF shape: {ddf.shape}")
        # STEP 5: 生成伪CT
        img = warp(t0, torch.from_numpy(ddf).float())
        imgs.append(img.detach().numpy())
    # STEP 6: 保存伪CT
    for index, img in enumerate(imgs):
        data = img[0, 0, :, :, :]
        print(type(data))
        print(data.shape)
        ct = t0_img.new_image_like(data)
        # STEP 7: 保存 nii
        print(f"Save {patient} T{index} ... ")
        ct.to_file(os.path.join(args.img_output_dir, f"{patient}_t{index}.nii"))
