from torch.utils.tensorboard import SummaryWriter
import os
import torch
import monai
import numpy as np
import ants
from nets import Net
from dataset import getDataLoader
from torch.nn import MSELoss, L1Loss
from monai.losses import MaskedLoss

torch.multiprocessing.set_sharing_strategy("file_system")
from monai.networks.utils import train_mode, eval_mode, save_state


torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
monai.utils.set_determinism(seed=0)
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--name", type=str, default="tnet", help="")
parser.add_argument("--B2A", action="store_true", default=False, help="")
parser.add_argument("--load", type=int, default=0, help="")
args = parser.parse_args()
image_loss = L1Loss()

load = args.load

output_image_path = f"./results/{args.name}/{load}/images"
output_ddf_path = f"./results/{args.name}/{load}/ddfs"
os.makedirs(output_image_path, exist_ok=True)
os.makedirs(output_ddf_path, exist_ok=True)

# DataLoader
val_loader, _ = getDataLoader(batch_size=1, num_workers=0, istry=False, mode="train")
# _,val_loader = getDataLoader(batch_size=1,num_workers=0,istry=False,mode="test")

net = Net()
print("# generator parameters:", sum(param.numel() for param in net.parameters()))
net = net.cuda()
gpus = [0, 1]
net = torch.nn.DataParallel(net, device_ids=gpus, output_device=gpus[0])

para_in_last_net = torch.load(f"./checkpoints/{args.name}/{args.load}.ckpt")
net.load_state_dict(para_in_last_net)


def save_nii(tx, epoch, filename, save_npy_path="./npy"):
    npy = tx.cpu().detach().numpy()  # [0, 0].transpose((1, 0, 2))
    np.save(os.path.join(save_npy_path, str(epoch) + "_" + filename), npy)


with eval_mode(net):
    epoch_loss_t1, epoch_loss_t2, epoch_loss_t3, epoch_loss_t4, epoch_loss_t5 = (
        0,
        0,
        0,
        0,
        0,
    )
    with torch.no_grad():
        step = 0
        for index, batch_data in enumerate(val_loader):
            step += 1
            t0_image = batch_data["t0_image"].cuda()
            results = net(t0_image)
            ddf01, ddf02, ddf03, ddf04, ddf05, t1, t2, t3, t4, t5 = results

            if not args.B2A:
                save_nii(
                    ddf01,
                    load,
                    batch_data["pid"][0] + "_ddf1",
                    save_npy_path=output_ddf_path,
                )
                save_nii(
                    ddf02,
                    load,
                    batch_data["pid"][0] + "_ddf2",
                    save_npy_path=output_ddf_path,
                )
                save_nii(
                    ddf03,
                    load,
                    batch_data["pid"][0] + "_ddf3",
                    save_npy_path=output_ddf_path,
                )
                save_nii(
                    ddf04,
                    load,
                    batch_data["pid"][0] + "_ddf4",
                    save_npy_path=output_ddf_path,
                )
                save_nii(
                    ddf05,
                    load,
                    batch_data["pid"][0] + "_ddf5",
                    save_npy_path=output_ddf_path,
                )

                save_nii(
                    t1,
                    load,
                    batch_data["pid"][0] + "_t1",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    t2,
                    load,
                    batch_data["pid"][0] + "_t2",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    t3,
                    load,
                    batch_data["pid"][0] + "_t3",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    t4,
                    load,
                    batch_data["pid"][0] + "_t4",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    t5,
                    load,
                    batch_data["pid"][0] + "_t5",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    batch_data["t0_image"],
                    load,
                    batch_data["pid"][0] + "_t0_real",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    batch_data["t1_image"],
                    load,
                    batch_data["pid"][0] + "_t1_real",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    batch_data["t2_image"],
                    load,
                    batch_data["pid"][0] + "_t2_real",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    batch_data["t3_image"],
                    load,
                    batch_data["pid"][0] + "_t3_real",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    batch_data["t4_image"],
                    load,
                    batch_data["pid"][0] + "_t4_real",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    batch_data["t5_image"],
                    load,
                    batch_data["pid"][0] + "_t5_real",
                    save_npy_path=output_image_path,
                )
            else:
                save_nii(
                    ddf01,
                    load,
                    batch_data["pid"][0] + "_ddf9",
                    save_npy_path=output_ddf_path,
                )
                save_nii(
                    ddf02,
                    load,
                    batch_data["pid"][0] + "_ddf8",
                    save_npy_path=output_ddf_path,
                )
                save_nii(
                    ddf03,
                    load,
                    batch_data["pid"][0] + "_ddf7",
                    save_npy_path=output_ddf_path,
                )
                save_nii(
                    ddf04,
                    load,
                    batch_data["pid"][0] + "_ddf6",
                    save_npy_path=output_ddf_path,
                )
                save_nii(
                    ddf05,
                    load,
                    batch_data["pid"][0] + "_ddf5",
                    save_npy_path=output_ddf_path,
                )
                save_nii(
                    t1,
                    load,
                    batch_data["pid"][0] + "_t9",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    t2,
                    load,
                    batch_data["pid"][0] + "_t8",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    t3,
                    load,
                    batch_data["pid"][0] + "_t7",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    t4,
                    load,
                    batch_data["pid"][0] + "_t6",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    t5,
                    load,
                    batch_data["pid"][0] + "_t5",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    batch_data["t9_image"],
                    load,
                    batch_data["pid"][0] + "_t9_real",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    batch_data["t8_image"],
                    load,
                    batch_data["pid"][0] + "_t8_real",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    batch_data["t7_image"],
                    load,
                    batch_data["pid"][0] + "_t7_real",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    batch_data["t6_image"],
                    load,
                    batch_data["pid"][0] + "_t6_real",
                    save_npy_path=output_image_path,
                )
                save_nii(
                    batch_data["t5_image"],
                    load,
                    batch_data["pid"][0] + "_t5_real",
                    save_npy_path=output_image_path,
                )
