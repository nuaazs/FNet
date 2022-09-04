from concurrent.futures import thread
import ants
import numpy as np
import os
from IPython import embed
from utils.ddf_tools import resample_3d_ddf
from tqdm import tqdm
import argparse
from multiprocessing.dummy import Pool as ThreadPool

def multi_get_nii(npys,thread_num):
    pool = ThreadPool(thread_num)
    pool.map(get_nii_tiny, npys)
    pool.close()
    pool.join()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net1_epoch', type=str, default='59',help='')
parser.add_argument('--net2_epoch', type=str, default='59',help='')
parser.add_argument('--net1_name', type=str, default='cbamunet_resunits2_A2B_ft',help='')
parser.add_argument('--net2_name', type=str, default='cbamunet_resunits2_B2A_ft',help='')
parser.add_argument('--save_path', type=str, default="./niis_59_from_npy",help='')
parser.add_argument('--size', type=str, default='128',help='')
parser.add_argument('--thread', type=int, default=20,help='')
parser.add_argument('--lung',action='store_true',default=False,help='')
args = parser.parse_args()

net1_npy_path = f"/mnt/zhaosheng/FNet/results/{args.net1_name}/{args.net1_epoch}/images"
net2_npy_path = f"/mnt/zhaosheng/FNet/results/{args.net2_name}/{args.net2_epoch}/images"
nii_save_path = args.save_path
os.makedirs(nii_save_path,exist_ok=True)
real_nii_path = "/mnt/zhaosheng/4dct/resampled"
npys = sorted([os.path.join(net1_npy_path,_file) for _file in  os.listdir(net1_npy_path) if "real" not in _file and "_t5" in _file])
print(f"Total: {len(npys)} Patients.")
# multi_get_nii(npys,args.thread)

for npy_file in tqdm(npys[:5]):
    p_index = npy_file.split("/")[-1].split(".")[0].split("_")[1]
    real_128_image_t0 = ants.image_read(os.path.join(real_nii_path,f"{p_index}_t0_resampled.nii"))
    real_128_image_t0.to_file(os.path.join(nii_save_path,f"{p_index}_t0_real.nii.gz"))
    real_128_image_t0.to_file(os.path.join(nii_save_path,f"{p_index}_t0_fake.nii.gz"))
    t0_npy = os.path.join(net1_npy_path,f"{args.net1_epoch}_{p_index}_t0_real.npy")
    t0_array = np.load(t0_npy,allow_pickle=True)[0,0]*3770 - 1000
    print(f"min {t0_array.min()} ,max {t0_array.max()}")
    for t in range(1,10):
        t_index = f"t{t}"
        if t<=5:
            real_npy = os.path.join(net1_npy_path,f"{args.net1_epoch}_{p_index}_{t_index}_real.npy")
            real_npy_array = np.load(real_npy,allow_pickle=True)[0,0]*3770 - 1000
            real_image_128 = real_128_image_t0.new_image_like(real_npy_array)
            real_image_128.to_file(os.path.join(nii_save_path,f"{p_index}_{t_index}_real.nii.gz"))
            print(f"real {t_index} min {real_npy_array.min()} ,max {real_npy_array.max()}")

            fake_npy = os.path.join(net1_npy_path,f"{args.net1_epoch}_{p_index}_{t_index}.npy")
            fake_npy_array = np.load(fake_npy,allow_pickle=True)[0,0]*3770 - 1000
            fake_image_128 = real_128_image_t0.new_image_like(fake_npy_array)
            fake_image_128.to_file(os.path.join(nii_save_path,f"{p_index}_{t_index}_fake.nii.gz"))
            print(f"fake {t_index} min {fake_npy_array.min()} ,max {fake_npy_array.max()}")
        
        else:
            real_npy = os.path.join(net2_npy_path,f"{args.net2_epoch}_{p_index}_{t_index}_real.npy")
            real_npy_array = np.load(real_npy,allow_pickle=True)[0,0]*3770 - 1000
            real_image_128 = real_128_image_t0.new_image_like(real_npy_array)
            real_image_128.to_file(os.path.join(nii_save_path,f"{p_index}_{t_index}_real.nii.gz"))

            fake_npy = os.path.join(net2_npy_path,f"{args.net2_epoch}_{p_index}_{t_index}.npy")
            fake_npy_array = np.load(fake_npy,allow_pickle=True)[0,0]*3770 - 1000
            fake_image_128 = real_128_image_t0.new_image_like(fake_npy_array)
            fake_image_128.to_file(os.path.join(nii_save_path,f"{p_index}_{t_index}_fake.nii.gz"))

