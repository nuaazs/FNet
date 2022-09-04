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
parser.add_argument('--save_path', type=str, default="./niis_from_real_ddf",help='')
args = parser.parse_args()
nii_128_save_path = args.save_path
ddf_path = "/mnt/zhaosheng/FNet/ddfs_nii"
real_nii_path = "/mnt/zhaosheng/4dct/resampled"
os.makedirs(nii_128_save_path,exist_ok=True)
pname_from_ddfs = sorted([pname.split("_")[1] for pname in os.listdir(ddf_path) if "0_5" in pname])
# pname_from_ddfs = sorted([ddf_path.split("_")[1] for _file in  os.path.listdir(ddf_path) if "nii" in _file])
print(pname_from_ddfs)
for p_index in pname_from_ddfs[:10]: # TODO: 只处理前44个
    
    real_128_image = ants.image_read(os.path.join(real_nii_path,f"{p_index}_t0_resampled.nii"))
    t0_128_torch = torch.tensor(real_128_image.numpy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
    spacing_512 = real_128_image.spacing
    shape_512 = real_128_image.shape
    real_128_image.to_file(os.path.join(nii_128_save_path,f"{p_index}_t0_fake.nii.gz"))

    for t in range(1,10):
        t_index = f"{t}"
        print(f"Now : {p_index} t{t}")
        
        file_path = os.path.join(ddf_path,f"394_{p_index}_0_{t_index}.nii")
        if os.path.exists(file_path):
            continue        
        fake_ddf_image_128 = ants.image_read(file_path)
        fake_image_128 = real_128_image.new_image_like(warp_layer(t0_128_torch, torch.tensor(fake_ddf_image_128.numpy()*(-1)).cuda().unsqueeze(dim=0)).cpu().detach().numpy()[0,0])
        fake_image_128.to_file(os.path.join(nii_128_save_path,f"{p_index}_t{t_index}_fake.nii.gz"))
