# coding = utf-8
# @Time    : 2022-10-17  01:16:12
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 筛选正常数据

import numpy as np
import preprocess.cfg as cfg
import os

def get_patients_withtumor():
    """获取有肿瘤的病人
    """
    tumor_pnames = [ _file.split("_")[0] for _file in os.listdir(cfg.tumor_path) if ".nrrd" in _file]
    print(f"Total #{len(tumor_pnames)} with tumor: \n\t{tumor_pnames}")
    return tumor_pnames
