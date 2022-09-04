import ants
import numpy as np
import os
from IPython import embed
from monai.networks.blocks import Warp
import torch
from IPython import embed
import argparse
from utils.ddf_tools import resample_3d_ddf
warp_layer = Warp()


parser = argparse.ArgumentParser(description='')
parser.add_argument('--net1_epoch', type=str, default='49',help='')
parser.add_argument('--net2_epoch', type=str, default='49',help='')
parser.add_argument('--net1_name', type=str, default='cbamunet_resunits2_A2B_ft',help='')
parser.add_argument('--net2_name', type=str, default='cbamunet_resunits2_B2A_ft',help='')
parser.add_argument('--save_path', type=str, default="./niis_from_unet",help='')
parser.add_argument('--size', type=str, default='128',help='')
parser.add_argument('--lung',action='store_true',default=False,help='')
args = parser.parse_args()

net1_npy_path = f"/mnt/zhaosheng/FNet/ddfs"
net2_npy_path = f"/mnt/zhaosheng/FNet/ddfs"

nii_128_save_path = args.save_path
os.makedirs(nii_128_save_path,exist_ok=True)

real_nii_path = "/mnt/zhaosheng/4dct/resampled"
print(net1_npy_path)
npys = sorted([os.path.join(net1_npy_path,_file) for _file in  os.listdir(net1_npy_path) if "_5_0.npy" in _file])
print(npys)

for npy_file in npys[:44]: # TODO: 只处理前44个
    p_index = npy_file.split("/")[-1].split(".")[0].split("_")[1]
    if os.path.exists(os.path.join(nii_128_save_path,f"{p_index}_t6_fake.nii.gz")):
        continue

    real_128_image = ants.image_read(os.path.join(real_nii_path,f"{p_index}_t0_resampled.nii"))
    t0_128_torch = torch.tensor(real_128_image.numpy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
    spacing_512 = real_128_image.spacing
    shape_512 = real_128_image.shape
    real_128_image.to_file(os.path.join(nii_128_save_path,f"{p_index}_t0_fake.nii.gz"))

    for t in range(1,10):
        t_index = f"{t}"
        print(f"Now : {p_index} t{t}")   
        if t<=5:
            if os.path.exists(os.path.join(nii_128_save_path,f"{p_index}_t{t_index}_fake.nii.gz")):
                continue
            fake_ddf_file = os.path.join(net1_npy_path,f"{args.net1_epoch}_{p_index}_{t_index}_0.npy")
            fake_ddf_array = np.load(fake_ddf_file,allow_pickle=True)
            print(fake_ddf_array.shape)
            fake_image_128 = real_128_image.new_image_like(warp_layer(t0_128_torch, torch.tensor(fake_ddf_array).cuda()).cpu().detach().numpy()[0,0])
            fake_image_128.to_file(os.path.join(nii_128_save_path,f"{p_index}_t{t_index}_fake.nii.gz"))

        else:
            if os.path.exists(os.path.join(nii_128_save_path,f"{p_index}_t{t_index}_fake.nii.gz")):
                continue
            fake_ddf_file = os.path.join(net2_npy_path,f"{args.net2_epoch}_{p_index}_{t_index}_0.npy")
            fake_ddf_array = np.load(fake_ddf_file,allow_pickle=True)
            fake_image_128 = real_128_image.new_image_like(warp_layer(t0_128_torch, torch.tensor(fake_ddf_array).cuda()).cpu().detach().numpy()[0,0])
            fake_image_128.to_file(os.path.join(nii_128_save_path,f"{p_index}_t{t_index}_fake.nii.gz"))
