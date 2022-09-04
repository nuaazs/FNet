import ants
import numpy as np
import os
from IPython import embed
from monai.networks.blocks import Warp
import torch
from IPython import embed
import argparse
warp_layer = Warp()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net1_epoch', type=str, default='14',help='')
parser.add_argument('--net2_epoch', type=str, default='14',help='')
parser.add_argument('--net1_name', type=str, default='cbamunet_resunits2_A2B_ft',help='')
parser.add_argument('--net2_name', type=str, default='cbamunet_resunits2_B2A_ft',help='')
parser.add_argument('--size', type=str, default='256',help='')
parser.add_argument('--lung',action='store_true',default=False,help='')


args = parser.parse_args()



def resample_3d_ddf(ddf,img):
    print(f" DDF shape: {ddf.shape}")
    print(f" IMG shape: {img.shape}")

    dist_shape = list(img.shape)
    print(f" dist_shape: {dist_shape}")
    ddf_result = []
    for i in range(3):
        now_ddf = ddf[i]
        fake_ddf_image_256 = ants.from_numpy(now_ddf)
        fake_ddf_image_512  = ants.resample_image(fake_ddf_image_256,dist_shape,True,4)
        
        ddf_result.append(fake_ddf_image_512.numpy()*4)    
    return np.array(ddf_result)

net1_npy_path = f"/mnt/zhaosheng/FNet/results/{args.net1_name}/{args.net1_epoch}/ddfs"
net2_npy_path = f"/mnt/zhaosheng/FNet/results/{args.net2_name}/{args.net2_epoch}/ddfs"

# 后处理nii保存地址
nii_512_save_path = "./niis_from_ddf_512"
nii_256_save_path = "./niis_from_ddf_512"
os.makedirs(nii_512_save_path,exist_ok=True)
os.makedirs(nii_256_save_path,exist_ok=True)

# 真实nii地址
real_nii_path = "/mnt/zhaosheng/4dct/resampled"
real_nii_path_512 = "/mnt/zhaosheng/4dct_data"

npys = sorted([os.path.join(net1_npy_path,_file) for _file in  os.listdir(net1_npy_path) if "ddf5" in _file])
print(npys)
for npy_file in npys[:44]: # TODO: 只处理前44个
    p_index = npy_file.split("/")[-1].split(".")[0].split("_")[1]
    if os.path.exists(os.path.join(nii_512_save_path,f"{p_index}_t6_fake.nii.gz")):
        continue

    real_512_image = ants.image_read(os.path.join(real_nii_path_512,f"t0/{p_index}_t0.nii"))
    t0_512_torch = torch.tensor(real_512_image.numpy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
    spacing_512 = real_512_image.spacing
    shape_512 = real_512_image.shape
    real_512_image.to_file(os.path.join(nii_512_save_path,f"{p_index}_t0_fake.nii.gz"))

    for t in range(1,10):
        t_index = f"{t}"
        print(f"Now : {p_index} t{t}")   
        if t<=5:
            if os.path.exists(os.path.join(nii_512_save_path,f"{p_index}_t{t_index}_fake.nii.gz")):
                continue
            fake_ddf_file = os.path.join(net1_npy_path,f"{args.net1_epoch}_{p_index}_ddf{t_index}.npy")
            fake_ddf_array = np.load(fake_ddf_file,allow_pickle=True)[0] # torch.tensor().cuda()
            fake_ddf_image_256 = ants.from_numpy(fake_ddf_array)
            fake_ddf_array_512  = resample_3d_ddf(fake_ddf_image_256,real_512_image)
            fake_image_512 = real_512_image.new_image_like(warp_layer(t0_512_torch, torch.tensor(fake_ddf_array_512).cuda().unsqueeze(dim=0)).cpu().detach().numpy()[0,0])
            fake_image_512.to_file(os.path.join(nii_512_save_path,f"{p_index}_t{t_index}_fake.nii.gz"))

        else:
            if os.path.exists(os.path.join(nii_512_save_path,f"{p_index}_t{t_index}_fake.nii.gz")):
                continue
            fake_ddf_file = os.path.join(net2_npy_path,f"{args.net2_epoch}_{p_index}_ddf{t_index}.npy")
            fake_ddf_array = np.load(fake_ddf_file,allow_pickle=True)[0] # torch.tensor().cuda()
            fake_ddf_image_256 = ants.from_numpy(fake_ddf_array)
            fake_ddf_array_512  = resample_3d_ddf(fake_ddf_image_256,real_512_image)
            fake_image_512 = real_512_image.new_image_like(warp_layer(t0_512_torch, torch.tensor(fake_ddf_array_512).cuda().unsqueeze(dim=0)).cpu().detach().numpy()[0,0])
            fake_image_512.to_file(os.path.join(nii_512_save_path,f"{p_index}_t{t_index}_fake.nii.gz"))
