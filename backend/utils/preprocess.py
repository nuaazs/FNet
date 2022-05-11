
# ██╗██╗███╗   ██╗████████╗
# ██║██║████╗  ██║╚══██╔══╝
# ██║██║██╔██╗ ██║   ██║
# ██║██║██║╚██╗██║   ██║
# ██║██║██║ ╚████║   ██║
# ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝
# @Time    : 2022-05-11 09:21:36.000-05:00
# @Author  : 𝕫𝕙𝕒𝕠𝕤𝕙𝕖𝕟𝕘
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : http://iint.icu/
# @File    : /home/zhaosheng/paper4/backend/utils/preprocess.py
# @Describe: Preprocess dicom files to tensor

import ants
import numpy as np
import torch

def preprocess_dcm(dcm_file_path):
    img = ants.image_read(dcm_file_path)
    npy = img.numpy()
    tensor_img = torch.Tensor(npy)
    print(tensor_img.shape)
    return tensor_img