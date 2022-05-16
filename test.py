from torch.utils.tensorboard import SummaryWriter
import os
import torch
import monai
import numpy as np
import ants
from nets import Net
from dataset import getDataLoader
from torch.nn import MSELoss,L1Loss
from monai.losses import MaskedLoss
torch.multiprocessing.set_sharing_strategy('file_system')
from monai.networks.utils import train_mode,eval_mode,save_state
import logging
import argparse
from utils import show_results,show_error,show_mask
from monai.data import Dataset, DataLoader, create_test_image_3d, decollate_batch
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandAffined,
    Resized,
    ScaleIntensityRanged,
    EnsureTyped,
    RandZoomd,
    RandRotate90d,
    RandFlipd,
    SaveImaged,
    Invertd,
    Activationsd,
    AsDiscreted
)

torch.cuda.empty_cache()
 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
monai.utils.set_determinism(seed=0)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, default="tnet",help='')

parser.add_argument('--istry', action='store_true', default=False,help='')
parser.add_argument('--B2A', action='store_true', default=False,help='')
parser.add_argument('--load', type=int, default=0,help='')
args = parser.parse_args()


load = args.load

os.makedirs(f"./outputs/{args.name}/{load}/images",exist_ok=True)
os.makedirs(f"./outputs/{args.name}/{load}/ddfs",exist_ok=True)

# DataLoader
val_loader,_ = getDataLoader(batch_size=1,num_workers=1,istry=args.istry)

net = Net()
print('# generator parameters:', sum(param.numel() for param in net.parameters()))
net = net.cuda()
gpus = [0,1]
net = torch.nn.DataParallel(net, device_ids=gpus, output_device=gpus[0])

para_in_last_net = torch.load(f"./checkpoints/{args.name}/{args.load}.ckpt")
net.load_state_dict(para_in_last_net)

def save_nii(tx,epoch,filename,save_npy_path="./npy"):
    npy = tx.cpu().detach().numpy()#[0, 0].transpose((1, 0, 2))
    np.save(os.path.join(save_npy_path,str(epoch)+"_"+filename),npy)


with eval_mode(net):
    with torch.no_grad():
        step = 0
        for index,batch_data in enumerate(val_loader):
            step += 1
            t0_image = batch_data["t0_image"].cuda()
            # _shape = batch_data["t0_image"].shape
            # _max = batch_data["t0_image"].max()
            # _min = batch_data["t0_image"].min()
            results = net(t0_image)
            ddf1,ddf2,ddf3,ddf4,ddf5 = results

            # ddf1,ddf2,ddf3,ddf4,ddf5,t1,t2,t3,t4,t5 = results
            # batch_data["fake_t1"] = t1
            # batch_data["fake_t2"] = t2
            # batch_data["fake_t3"] = t3
            # batch_data["fake_t4"] = t4
            # batch_data["fake_t5"] = t5

            batch_data["ddf1"] = ddf1
            batch_data["ddf2"] = ddf2
            batch_data["ddf3"] = ddf3
            batch_data["ddf4"] = ddf4
            batch_data["ddf5"] = ddf5
            
            if not args.B2A:
                save_nii(ddf1,load,batch_data["pid"][0]+'_ddf1',save_npy_path=f"./outputs/{args.name}/{load}/ddfs")
                save_nii(ddf2,load,batch_data["pid"][0]+'_ddf2',save_npy_path=f"./outputs/{args.name}/{load}/ddfs")
                save_nii(ddf3,load,batch_data["pid"][0]+'_ddf3',save_npy_path=f"./outputs/{args.name}/{load}/ddfs")
                save_nii(ddf4,load,batch_data["pid"][0]+'_ddf4',save_npy_path=f"./outputs/{args.name}/{load}/ddfs")
                save_nii(ddf5,load,batch_data["pid"][0]+'_ddf5',save_npy_path=f"./outputs/{args.name}/{load}/ddfs")
            else:
                save_nii(ddf1,load,batch_data["pid"][0]+'_ddf9',save_npy_path=f"./outputs/{args.name}/{load}/ddfs")
                save_nii(ddf2,load,batch_data["pid"][0]+'_ddf8',save_npy_path=f"./outputs/{args.name}/{load}/ddfs")
                save_nii(ddf3,load,batch_data["pid"][0]+'_ddf7',save_npy_path=f"./outputs/{args.name}/{load}/ddfs")
                save_nii(ddf4,load,batch_data["pid"][0]+'_ddf6',save_npy_path=f"./outputs/{args.name}/{load}/ddfs")
                save_nii(ddf5,load,batch_data["pid"][0]+'_ddf5',save_npy_path=f"./outputs/{args.name}/{load}/ddfs")
