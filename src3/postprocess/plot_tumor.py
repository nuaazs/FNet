# coding = utf-8
# @Time    : 2022-10-17  01:27:03
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 可视化肿瘤位置.

import numpy as np
import ants
import os
from IPython import embed
from monai.networks.blocks import Warp
import torch
import matplotlib.pyplot as plt
import cv2

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

def normalize(array):
    """_summary_: 归一化

    Args:
        array (nparray): 输入矩阵

    Returns:
        nparray: 归一化后的矩阵
    """
    array = array - np.min(array)
    array = array / np.max(array)
    return array

def plot_ct_mask(ct, mask):
    """_summary_: 叠加显示CT和肿瘤

    Args:
        ct (nparray): ct矩阵,2d,128*128
        mask (nparray): mask矩阵,2d,128*128
    """
    # STEP 1 npy转cv2图片
    # embed()
    ct = normalize(ct)

    ct_img = np.array(ct * 255, dtype = np.uint8)
    mask_img = np.array(mask * 255, dtype = np.uint8)

    ct_img = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2BGR)

    # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    threshed = cv2.adaptiveThreshold(mask_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

    # print(ct.max(), ct.min())
    # print("start plot")
    # #ct_img = cv2.cvtColor(ct*256,cv2.COLOR_GRAY2BGR)
    # #ct_img = cv2.cvtColor(ct_img,cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("ct.png",np.int8(ct*255))
    # #mask_img = cv2.cvtColor(mask*256,cv2.COLOR_GRAY2BGR)
    # #mask_img = cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("mask.png",np.int8(mask*255))

    # ct_img = cv2.imread("ct.png").astype(np.uint8)
    # mask_img = cv2.imread("mask.png").astype(np.uint8)
    # print(type(mask_img))
    # mask_img = np.int8(mask_img)
    # print(type(mask_img))
    
    ret, thresh = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ct_img, contours, -1, (0, 255, 0), 1)

    cv2.imwrite("ct_img_with_tumor.png",ct_img)
    # cv2.namedWindow("ct")
    # cv2.imshow("ct", ct_img)
    # cv2.waitKey(0)

def plot_loc(ddf_real,ddf_fake,ct, mask,pname,tumor_loc,save_path):
    ppath = os.path.join(save_path,pname)
    os.makedirs(os.path.join(save_path,pname,"ct"),exist_ok=True)
    os.makedirs(os.path.join(save_path,pname,"x"),exist_ok=True)
    os.makedirs(os.path.join(save_path,pname,"y"),exist_ok=True)
    os.makedirs(os.path.join(save_path,pname,"z"),exist_ok=True)
    os.makedirs(os.path.join(save_path,pname,"ddf"),exist_ok=True)
    plot_slices = [int(tumor_loc[2])]# [45,32,25,17]
    raw_mask = torch.FloatTensor(mask.numpy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0) # 1,1,128,128,64
    raw_ct = torch.FloatTensor(ct.numpy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0) # 1,1,128,128,64
    ddf_real = torch.FloatTensor(ddf_real).cuda().unsqueeze(dim=0) # 1,3,128,128,64
    ddf_fake = torch.FloatTensor(ddf_fake).cuda().unsqueeze(dim=0) # 1,3,128,128,64

    tumor_real = warp_layer(raw_mask, ddf_real).cpu().detach().numpy()
    tumor_fake = warp_layer(raw_mask, ddf_fake).cpu().detach().numpy()
    ct_real = warp_layer(raw_ct, ddf_real)#.cpu().detach().numpy()
    ct_fake = warp_layer(raw_ct, ddf_fake)#.cpu().detach().numpy()
    print("start")
    plot_ct_mask(ct_real.detach().cpu().numpy()[0,0,:,int(tumor_loc[1]),:],tumor_real[0,0,:,int(tumor_loc[1]),:])
    plot_ct_mask(ct_fake.detach().cpu().numpy()[0,0,:,int(tumor_loc[1]),:],tumor_fake[0,0,:,int(tumor_loc[1]),:])
    # for slice in plot_slices:
    #     plt.figure()
    #     plt.subplot(1,3,1)
    #     plt.imshow(raw_ct.detach().cpu().numpy()[0,0,:,:,slice])
    #     plt.title("ct_input")
    #     plt.subplot(1,3,2)
    #     plt.imshow(ct_real[0,0,:,:,slice])
    #     plt.title("ct_real")
    #     plt.subplot(1,3,3)
    #     plt.imshow(ct_fake[0,0,:,:,slice])
    #     plt.title("ct_fake")
    #     plt.savefig(os.path.join(ppath,"ct",f"{slice}.png"))

    #     # X
    #     plt.figure(figsize=(10,10),dpi=300)
    #     plt.subplot(3,3,1)
    #     plt.imshow(raw_mask.detach().cpu().numpy()[0,0,int(tumor_loc[0])-1,:,:])
    #     plt.title("tumor_t0 - 1")
    #     plt.subplot(3,3,2)
    #     plt.imshow(tumor_real[0,0,int(tumor_loc[0])-1,:,:])
    #     plt.title("tumor_real - 1")
    #     plt.subplot(3,3,3)
    #     plt.imshow(tumor_fake[0,0,int(tumor_loc[0])-1,:,:])
    #     plt.title("tumor_fake - 1")

    #     plt.subplot(3,3,4)
    #     plt.imshow(raw_mask.detach().cpu().numpy()[0,0,int(tumor_loc[0]),:,:])
    #     plt.title("tumor_t0")
    #     plt.subplot(3,3,5)
    #     plt.imshow(tumor_real[0,0,int(tumor_loc[0]),:,:])
    #     plt.title("tumor_real")
    #     plt.subplot(3,3,6)
    #     plt.imshow(tumor_fake[0,0,int(tumor_loc[0]),:,:])
    #     plt.title("tumor_fake")
        
    #     plt.subplot(3,3,7)
    #     plt.imshow(raw_mask.detach().cpu().numpy()[0,0,int(tumor_loc[0])+1,:,:])
    #     plt.title("tumor_t0 + 1")
    #     plt.subplot(3,3,8)
    #     plt.imshow(tumor_real[0,0,int(tumor_loc[0])+1,:,:])
    #     plt.title("tumor_real + 1")
    #     plt.subplot(3,3,9)
    #     plt.imshow(tumor_fake[0,0,int(tumor_loc[0])+1,:,:])
    #     plt.title("tumor_fake + 1")
    #     plt.savefig(os.path.join(ppath,"x",f"{slice}.png"))



    #     # Y 方向
    #     plt.figure(figsize=(10,10),dpi=300)
    #     plt.subplot(3,3,1)
    #     plt.imshow(raw_mask.detach().cpu().numpy()[0,0,:,int(tumor_loc[1])-1,:])
    #     plt.title("tumor_t0 - 1")
    #     plt.subplot(3,3,2)
    #     plt.imshow(tumor_real[0,0,:,int(tumor_loc[1])-1,:])
    #     plt.title("tumor_real - 1")
    #     plt.subplot(3,3,3)
    #     plt.imshow(tumor_fake[0,0,:,int(tumor_loc[1])-1,:])
    #     plt.title("tumor_fake - 1")

    #     plt.subplot(3,3,4)
    #     plt.imshow(raw_mask.detach().cpu().numpy()[0,0,:,int(tumor_loc[1]),:])
    #     plt.title("tumor_t0")
    #     plt.subplot(3,3,5)
    #     plt.imshow(tumor_real[0,0,:,int(tumor_loc[1]),:])
    #     plt.title("tumor_real")
    #     plt.subplot(3,3,6)
    #     plt.imshow(tumor_fake[0,0,:,int(tumor_loc[1]),:])
    #     plt.title("tumor_fake")
        
    #     plt.subplot(3,3,7)
    #     plt.imshow(raw_mask.detach().cpu().numpy()[0,0,:,int(tumor_loc[1])+1,:])
    #     plt.title("tumor_t0 + 1")
    #     plt.subplot(3,3,8)
    #     plt.imshow(tumor_real[0,0,:,int(tumor_loc[1])+1,:])
    #     plt.title("tumor_real + 1")
    #     plt.subplot(3,3,9)
    #     plt.imshow(tumor_fake[0,0,:,int(tumor_loc[1])+1,:])
    #     plt.title("tumor_fake + 1")
    #     plt.savefig(os.path.join(ppath,"y",f"{slice}.png"))



    #     # Z 方向
    #     plt.figure(figsize=(10,10),dpi=300)
    #     plt.subplot(3,3,1)
    #     plt.imshow(raw_mask.detach().cpu().numpy()[0,0,:,:,int(tumor_loc[2])-1])
    #     plt.title("tumor_t0 - 1")
    #     plt.subplot(3,3,2)
    #     plt.imshow(tumor_real[0,0,:,:,int(tumor_loc[2])-1])
    #     plt.title("tumor_real - 1")
    #     plt.subplot(3,3,3)
    #     plt.imshow(tumor_fake[0,0,:,:,int(tumor_loc[2])-1])
    #     plt.title("tumor_fake - 1")

    #     plt.subplot(3,3,4)
    #     plt.imshow(raw_mask.detach().cpu().numpy()[0,0,:,:,int(tumor_loc[2])])
    #     plt.title("tumor_t0")
    #     plt.subplot(3,3,5)
    #     plt.imshow(tumor_real[0,0,:,:,int(tumor_loc[2])])
    #     plt.title("tumor_real")
    #     plt.subplot(3,3,6)
    #     plt.imshow(tumor_fake[0,0,:,:,int(tumor_loc[2])])
    #     plt.title("tumor_fake")
        
    #     plt.subplot(3,3,7)
    #     plt.imshow(raw_mask.detach().cpu().numpy()[0,0,:,:,int(tumor_loc[2])+1])
    #     plt.title("tumor_t0 + 1")
    #     plt.subplot(3,3,8)
    #     plt.imshow(tumor_real[0,0,:,:,int(tumor_loc[2])+1])
    #     plt.title("tumor_real + 1")
    #     plt.subplot(3,3,9)
    #     plt.imshow(tumor_fake[0,0,:,:,int(tumor_loc[2])+1])
    #     plt.title("tumor_fake + 1")
    #     plt.savefig(os.path.join(ppath,"z",f"{slice}.png"))



    #     # plt.subplot(3,3,7)
    #     # plt.imshow(raw_mask[0,0,:,:,slice])
    #     # plt.title("tumor_t0")
    #     plt.figure()
    #     plt.subplot(1,3,2)
    #     plt.imshow(ddf_fake.detach().cpu().numpy()[0,0,:,:,2])
    #     plt.colorbar()
    #     plt.title("ddf_fake")
    #     plt.subplot(1,3,3)
    #     plt.imshow(ddf_real.detach().cpu().numpy()[0,0,:,:,2])
    #     plt.colorbar()
    #     plt.title("ddf_fake")

    #     plt.savefig(os.path.join(ppath,"ddf",f"{slice}.png"))




if __name__ == "__main__":
    warp_layer = Warp()

    fake_images = get_result(output_path, EPOCH, "_image_fake")
    real_images = get_result(output_path, EPOCH, "_image_real")
    fake_ddfs = get_result(output_path, EPOCH, "_ddf_fake")
    real_ddfs = get_result(output_path, EPOCH, "_ddf_real")

    error = []
    for index in range(len(real_ddfs)):
        pname = fake_images[index].split("/")[-1].split("_")[1]
        print(f"**{pname}")
        ddfs_fake = np.load(
            os.path.join(output_path, f"{EPOCH}_{pname}_ddf_fake.npy")
        ).reshape((9, 3, 128, 128, 64))
        ddfs_real = np.load(
            os.path.join(output_path, f"{EPOCH}_{pname}_ddf_real.npy")
        ).reshape((9, 3, 128, 128, 64))
        ct = ants.image_read(os.path.join(real_nii_path, f"{pname}_t0.nii"))
        mask = ants.image_read(
            os.path.join(tumor_path, f"{pname}_t0_Segmentation.seg.nrrd")
        )  # .numpy()
        i = 5

        print(f"\t#T{i}")
        ddf_fake = ddfs_fake[i]
        ddf_real = ddfs_real[i]
        tumor_loc = getLoc(mask.numpy())
        print(f"Tumor loc: {tumor_loc}")
        plot_loc(ddf_real,ddf_fake, ct, mask,pname,tumor_loc,save_path="./pngs/")
        # break
