# coding = utf-8
# @Time    : 2022-09-21  07:47:25
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Get t5 from t0 by unet.

from torch.utils.tensorboard import SummaryWriter
import os
import torch
import monai
import numpy as np
from nets import FNet
from dataset import getDataLoader
from torch.nn import MSELoss, L1Loss
from monai.losses import MaskedLoss
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
from monai.networks.utils import train_mode, eval_mode, save_state
import argparse
from utils.save import save_nii
import random

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

seed = 123456
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
monai.utils.set_determinism(seed=seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--name", type=str, default="fnet0904", help="")
parser.add_argument(
    "--ddf_dir", type=str, default="/mnt/zhaosheng/FNet/data/ddfs", help=""
)
parser.add_argument(
    "--img_dir", type=str, default="/mnt/zhaosheng/4dct/resampled", help=""
)
parser.add_argument("--ddf_prefix", type=str, default="49", help="")
parser.add_argument("--batch_size", type=int, default=2, help="")
parser.add_argument("--num_workers", type=int, default=0, help="")
parser.add_argument("--epochs", type=int, default=200, help="")
parser.add_argument("--val_interval", type=int, default=1, help="")
parser.add_argument("--save_interval", type=int, default=20, help="")
parser.add_argument("--save_npy_interval", type=int, default=20, help="")
parser.add_argument("--istry", action="store_true", default=False, help="")
parser.add_argument("--load", type=int, default=0, help="")

args = parser.parse_args()

os.makedirs(f"./runs/{args.name}", exist_ok=True)
os.makedirs(f"./checkpoints/{args.name}/", exist_ok=True)
os.makedirs(f"./results/{args.name}/", exist_ok=True)

writer = SummaryWriter(f"./runs/{args.name}")
train_loader, val_loader = getDataLoader(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    istry=args.istry,
    ddf_prefix=args.ddf_prefix,
    ddf_dir=args.ddf_dir,
    img_dir=args.img_dir,
)
net = FNet()
print("# generator parameters:", sum(param.numel() for param in net.parameters()))
net = net.cuda()
gpus = [0, 1]
net = torch.nn.DataParallel(net, device_ids=gpus, output_device=gpus[0])

image_loss = L1Loss()
mask_loss = MaskedLoss(L1Loss)
optimizer = torch.optim.Adam(net.parameters(), 1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=10, factor=0.1, min_lr=1e-6
)

if args.load > 0:
    para_in_last_net = torch.load(f"./checkpoints/{args.name}/{args.load}.ckpt")
    net.load_state_dict(para_in_last_net)

for epoch in range(args.load, args.load + args.epochs):
    print(f"epoch {epoch + 1}/{args.load+args.epochs}")
    with train_mode(net):
        total_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            optimizer.zero_grad()
            t0 = batch_data["t0"].cuda()
            ddf, t5 = net(t0)
            loss = image_loss(t5, batch_data["t5"].cuda())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        total_loss /= step
        writer.add_scalar(
            "learning rate",
            optimizer.state_dict()["param_groups"][0]["lr"],
            global_step=epoch,
            walltime=None,
        )
        writer.add_scalar("train_loss", total_loss, global_step=epoch, walltime=None)
        print(f"loss: {total_loss:.4f}")

    if (epoch + 1) % args.save_interval == 0:
        save_state(net.state_dict(), f"./checkpoints/{args.name}/{epoch}.ckpt")

    if (epoch + 1) % args.val_interval == 0:
        with eval_mode(net):
            with torch.no_grad():
                total_loss = 0
                step = 0
                for batch_data in val_loader:
                    step += 1
                    t0 = batch_data["t0"].cuda()
                    ddf, t5 = net(t0)
                    loss = image_loss(t5, batch_data["t5"].cuda())
                    total_loss += loss.item()
                if (epoch + 1) % args.save_npy_interval == 0:
                    save_nii(
                        tensor=ddf,
                        filename=str(epoch) + "_" + batch_data["pid"][0] + "_ddf_fake",
                        save_npy_path=f"./results/{args.name}/",
                    )
                    save_nii(
                        tensor=batch_data["ddf"],
                        filename=str(epoch) + "_" + batch_data["pid"][0] + "_ddf_real",
                        save_npy_path=f"./results/{args.name}/",
                    )
                    save_nii(
                        tensor=t5,
                        filename=str(epoch)
                        + "_"
                        + batch_data["pid"][0]
                        + "_image_fake",
                        save_npy_path=f"./results/{args.name}/",
                    )
                    save_nii(
                        tensor=batch_data["t5"],
                        filename=str(epoch)
                        + "_"
                        + batch_data["pid"][0]
                        + "_image_real",
                        save_npy_path=f"./results/{args.name}/",
                    )
                    save_nii(
                        tensor=batch_data["t0"],
                        filename=str(epoch)
                        + "_"
                        + batch_data["pid"][0]
                        + "_image_input",
                        save_npy_path=f"./results/{args.name}/",
                    )
                total_loss /= step
                writer.add_scalar(
                    "learning rate",
                    optimizer.state_dict()["param_groups"][0]["lr"],
                    global_step=epoch,
                    walltime=None,
                )
                writer.add_scalar(
                    "val_loss", total_loss, global_step=epoch, walltime=None
                )
                print(f"Val loss: {total_loss:.4f} \n\n")
