# coding = utf-8
# @Time    : 2022-10-17  01:16:12
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 计算肿瘤部位平均偏移量以及误差值.

import numpy as np
import ants
import os
from IPython import embed
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# cfg
from cfg import real_nii_path
from cfg import tumor_path
from cfg import output_path
from cfg import EPOCH


def getLoc(_npy):
    """_summary_: 获取肿瘤中心点坐标

    Args:
        _npy (nparray): mask矩阵

    Returns:
        set: 肿瘤中心坐标
    """
    _npy = np.array(_npy)
    loc = np.mean(np.argwhere(_npy == 1), axis=0)
    return loc


def get_result(output_path, epoch, type="_ddf_fake"):
    """_summary_: 获取文件列表

    Args:
        output_path (string): 结果保存路径
        epoch (int): epoch数
        type (str, optional): 结果类型. Defaults to "_ddf_fake".

    Returns:
        list: 文件列表
    """
    return sorted(
        [
            os.path.join(output_path, _file)
            for _file in os.listdir(output_path)
            if _file.startswith(f"{epoch}_") and str(type) in _file
        ],
        key=lambda x: x.split("_")[1],
    )


def get_loc(ddf, ct, mask):
    """获取肿瘤部位平均偏移量

    Args:
        ddf (nparray): ddf矩阵
        ct (nparray): ct矩阵
        mask (nparray): 肿瘤mask矩阵

    Returns:
        set: 肿瘤部位平均偏移量
    """
    assert ct.shape == mask.shape
    # print(mask.shape)
    # # mean_loc = np.mean(mask,axis\\\)
    # # print(mean_loc)

    # plt.figure()
    # plt.imshow(mask[:,int(getLoc(mask)[1]),:])
    # plt.savefig("temp.png")
    # plt.show()

    # plt.figure()
    # plt.imshow(ct[:,int(getLoc(mask)[1]),:])
    # plt.savefig("temp2.png")
    # plt.show()

    ddf_x = ddf[0]
    ddf_y = ddf[1]
    ddf_z = ddf[2]

    ddf_x[mask < 0.5] = 0
    ddf_y[mask < 0.5] = 0
    ddf_z[mask < 0.5] = 0
    x_mean = np.mean(ddf_x[mask > 0.5])
    y_mean = np.mean(ddf_y[mask > 0.5])
    z_mean = np.mean(ddf_z[mask > 0.5])
    return x_mean, y_mean, z_mean


if __name__ == "__main__":
    fake_images = get_result(output_path, EPOCH, "_image_fake")
    real_images = get_result(output_path, EPOCH, "_image_real")
    fake_ddfs = get_result(output_path, EPOCH, "_ddf_fake")
    real_ddfs = get_result(output_path, EPOCH, "_ddf_real")

    error = []  # 相对误差
    trans_real = []  # Real
    trans_fake = []  # Fake

    for index in range(len(real_ddfs)):
        pname = fake_images[index].split("/")[-1].split("_")[1]
        print(f"**{pname}")
        ddfs_fake = np.load(
            os.path.join(output_path, f"{EPOCH}_{pname}_ddf_fake.npy")
        ).reshape((9, 3, 128, 128, 64))
        ddfs_real = np.load(
            os.path.join(output_path, f"{EPOCH}_{pname}_ddf_real.npy")
        ).reshape((9, 3, 128, 128, 64))
        ct = ants.image_read(os.path.join(real_nii_path, f"{pname}_t0.nii")).numpy()
        mask = ants.image_read(
            os.path.join(tumor_path, f"{pname}_t0_Segmentation.seg.nrrd")
        ).numpy()

        error_list = []
        trans_fake_list = []
        trans_real_list = []

        for i in range(9):
            print(f"\t#T{i}")
            ddf_fake = ddfs_fake[i]
            ddf_real = ddfs_real[i]
            x_mean_fake, y_mean_fake, z_mean_fake = get_loc(ddf_fake, ct, mask)
            print(f"\t\tFake:{x_mean_fake},{y_mean_fake},{z_mean_fake}")
            x_mean_real, y_mean_real, z_mean_real = get_loc(ddf_real, ct, mask)
            print(f"\t\tReal:{x_mean_real},{y_mean_real},{z_mean_real}")
            x_err = (abs(x_mean_real) - abs(x_mean_fake)) / abs(x_mean_real)
            y_err = (abs(y_mean_real) - abs(y_mean_fake)) / abs(y_mean_real)
            z_err = (abs(z_mean_real) - abs(z_mean_fake)) / abs(z_mean_real)
            error_list.append([x_err, y_err, z_err])
            trans_real_list.append([x_mean_real, y_mean_real, z_mean_real])
            trans_fake_list.append([x_mean_fake, y_mean_fake, z_mean_fake])
        error.append(error_list)
        trans_real.append(trans_real_list)
        trans_fake.append(trans_fake_list)
    error = np.abs(np.array(error))
    trans_real = np.abs(np.array(trans_real))
    trans_fake = np.abs(np.array(trans_fake))

    error_1 = np.mean(error, axis=0)
    error_2 = np.mean(error_1, axis=0)
    print(f"误差:\t{error_2}")

    real_1 = np.max(trans_real, axis=0)
    real_2 = np.max(real_1, axis=0)
    print(f"Real:\t{real_2}")

    fake_1 = np.max(trans_fake, axis=0)
    fake_2 = np.max(fake_1, axis=0)
    print(f"Fake:\t{fake_2}")
    embed()
