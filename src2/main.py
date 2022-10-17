import os
import torch
import monai
import numpy as np
import logging
import argparse
import random
import torchsummary

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
from torch.nn import MSELoss, L1Loss
from monai.losses import MaskedLoss
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy("file_system")
from monai.networks.utils import train_mode, eval_mode, save_state

# utils
from utils.save import save_nii
from nets import FNet
from dataset import getDataLoader

# seed
seed = 123456
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
monai.utils.set_determinism(seed=seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# args
parser = argparse.ArgumentParser(description="")
parser.add_argument("--name", type=str, default="fnet0904", help="")
parser.add_argument(
    "--img_dir", type=str, default="/mnt/zhaosheng/4dct/resampled", help=""
)
parser.add_argument("--ddf_prefix", type=str, default="49", help="")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--num_workers", type=int, default=0, help="")
parser.add_argument("--epochs", type=int, default=200, help="")
parser.add_argument("--val_interval", type=int, default=1, help="")
parser.add_argument("--save_interval", type=int, default=20, help="")
parser.add_argument("--save_npy_interval", type=int, default=20, help="")
parser.add_argument("--istry", action="store_true", default=False, help="")
parser.add_argument("--load", type=int, default=0, help="")
args = parser.parse_args()

# make dirs
os.makedirs(f"./runs/{args.name}", exist_ok=True)
os.makedirs(f"./checkpoints/{args.name}/", exist_ok=True)
os.makedirs(f"./results/{args.name}/", exist_ok=True)

writer = SummaryWriter(f"./runs/{args.name}")
train_loader, val_loader = getDataLoader(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    istry=args.istry,
    img_dir=args.img_dir,
)
net = FNet()
# print("# generator parameters:", sum(param.numel() for param in net.parameters()))
# torchsummary.summary(net.cuda(), (1,128,128,64))

net = net.cuda()
gpus = [0, 1]
net = torch.nn.DataParallel(net, device_ids=gpus, output_device=gpus[0])


image_loss = L1Loss()
ddf_loss = L1Loss()
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
        _loss_1 = 0
        _loss_2 = 0
        _loss_3 = 0
        _loss_4 = 0
        _loss_5 = 0
        _loss_6 = 0
        _loss_7 = 0
        _loss_8 = 0
        _loss_9 = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            optimizer.zero_grad()
            t0_image = batch_data["t0_image"].cuda()
            results = net(t0_image)
            (
                image1,
                image2,
                image3,
                image4,
                image5,
                image6,
                image7,
                image8,
                image9,
                ddf1,
                ddf2,
                ddf3,
                ddf4,
                ddf5,
                ddf6,
                ddf7,
                ddf8,
                ddf9,
            ) = results

            loss_1 = image_loss(image1, batch_data["t1_image"].cuda())
            loss_2 = image_loss(image2, batch_data["t2_image"].cuda())
            loss_3 = image_loss(image3, batch_data["t3_image"].cuda())
            loss_4 = image_loss(image4, batch_data["t4_image"].cuda())
            loss_5 = image_loss(image5, batch_data["t5_image"].cuda())
            loss_6 = image_loss(image6, batch_data["t6_image"].cuda())
            loss_7 = image_loss(image7, batch_data["t7_image"].cuda())
            loss_8 = image_loss(image8, batch_data["t8_image"].cuda())
            loss_9 = image_loss(image9, batch_data["t9_image"].cuda())

            ddf_1 = ddf_loss(ddf1, batch_data["ddf1"].cuda())
            ddf_2 = ddf_loss(ddf2, batch_data["ddf2"].cuda())
            ddf_3 = ddf_loss(ddf3, batch_data["ddf3"].cuda())
            ddf_4 = ddf_loss(ddf4, batch_data["ddf4"].cuda())
            ddf_5 = ddf_loss(ddf5, batch_data["ddf5"].cuda())
            ddf_6 = ddf_loss(ddf6, batch_data["ddf6"].cuda())
            ddf_7 = ddf_loss(ddf7, batch_data["ddf7"].cuda())
            ddf_8 = ddf_loss(ddf8, batch_data["ddf8"].cuda())
            ddf_9 = ddf_loss(ddf9, batch_data["ddf9"].cuda())

            _loss_1 += loss_1.item()
            _loss_2 += loss_2.item()
            _loss_3 += loss_3.item()
            _loss_4 += loss_4.item()
            _loss_5 += loss_5.item()
            _loss_6 += loss_6.item()
            _loss_7 += loss_7.item()
            _loss_8 += loss_8.item()
            _loss_9 += loss_9.item()
            loss = (
                loss_1
                + loss_2
                + loss_3
                + loss_4
                + loss_5
                + loss_6
                + loss_7
                + loss_8
                + loss_9
            )
            loss.backward()
            optimizer.step()

        _loss_1 /= step
        _loss_2 /= step
        _loss_3 /= step
        _loss_4 /= step
        _loss_5 /= step
        _loss_6 /= step
        _loss_7 /= step
        _loss_8 /= step
        _loss_9 /= step

        total_loss_train = (
            _loss_1
            + _loss_2
            + _loss_3
            + _loss_4
            + _loss_5
            + _loss_6
            + _loss_7
            + _loss_8
            + _loss_9
        )

        writer.add_scalar(
            "learning rate",
            optimizer.state_dict()["param_groups"][0]["lr"],
            global_step=epoch,
            walltime=None,
        )
        writer.add_scalar("t1_train", _loss_1, global_step=epoch, walltime=None)
        writer.add_scalar("t2_train", _loss_2, global_step=epoch, walltime=None)
        writer.add_scalar("t3_train", _loss_3, global_step=epoch, walltime=None)
        writer.add_scalar("t4_train", _loss_4, global_step=epoch, walltime=None)
        writer.add_scalar("t5_train", _loss_5, global_step=epoch, walltime=None)
        writer.add_scalar("t6_train", _loss_6, global_step=epoch, walltime=None)
        writer.add_scalar("t7_train", _loss_7, global_step=epoch, walltime=None)
        writer.add_scalar("t8_train", _loss_8, global_step=epoch, walltime=None)
        writer.add_scalar("t9_train", _loss_9, global_step=epoch, walltime=None)
        writer.add_scalar(
            "total_loss_train", total_loss_train, global_step=epoch, walltime=None
        )
        print(
            f"DDF Loss: \n\tloss t1: {_loss_1:.4f}\n\tloss t2: {_loss_2:.4f}\n\tloss t3: {_loss_3:.4f}\n\tloss t4: {_loss_4:.4f}\n\tloss t5: {_loss_5:.4f}\n\tloss t6: {_loss_6:.4f}\n\tloss t7: {_loss_7:.4f}\n\tloss t8: {_loss_8:.4f}\n\tloss t9: {_loss_9:.4f}"
        )
        print(f"Total Loss:{total_loss_train}")

    if (epoch + 1) % args.save_interval == 0:
        save_state(net.state_dict(), f"./checkpoints/{args.name}/{epoch}.ckpt")

    if (epoch + 1) % args.val_interval == 0:
        with eval_mode(net):
            with torch.no_grad():
                _loss_1 = 0
                _loss_2 = 0
                _loss_3 = 0
                _loss_4 = 0
                _loss_5 = 0
                _loss_6 = 0
                _loss_7 = 0
                _loss_8 = 0
                _loss_9 = 0
                step = 0
                for batch_data in val_loader:
                    step += 1
                    t0_image = batch_data["t0_image"].cuda()
                    results = net(t0_image)
                    (
                        image1,
                        image2,
                        image3,
                        image4,
                        image5,
                        image6,
                        image7,
                        image8,
                        image9,
                    ) = results

                    loss_1 = image_loss(image1, batch_data["t1_image"].cuda())
                    loss_2 = image_loss(image2, batch_data["t2_image"].cuda())
                    loss_3 = image_loss(image3, batch_data["t3_image"].cuda())
                    loss_4 = image_loss(image4, batch_data["t4_image"].cuda())
                    loss_5 = image_loss(image5, batch_data["t5_image"].cuda())
                    loss_6 = image_loss(image6, batch_data["t6_image"].cuda())
                    loss_7 = image_loss(image7, batch_data["t7_image"].cuda())
                    loss_8 = image_loss(image8, batch_data["t8_image"].cuda())
                    loss_9 = image_loss(image9, batch_data["t9_image"].cuda())

                    _loss_1 += loss_1.item()
                    _loss_2 += loss_2.item()
                    _loss_3 += loss_3.item()
                    _loss_4 += loss_4.item()
                    _loss_5 += loss_5.item()
                    _loss_6 += loss_6.item()
                    _loss_7 += loss_7.item()
                    _loss_8 += loss_8.item()
                    _loss_9 += loss_9.item()

                    if (epoch + 1) % args.save_npy_interval == 0:
                        for index, image in enumerate(results, 1):
                            save_nii(
                                tensor=image,
                                filename=str(epoch)
                                + "_"
                                + batch_data["pid"][0]
                                + f"_t{index}_fake",
                                save_npy_path=f"./results/{args.name}/",
                            )

                _loss_1 /= step
                _loss_2 /= step
                _loss_3 /= step
                _loss_4 /= step
                _loss_5 /= step
                _loss_6 /= step
                _loss_7 /= step
                _loss_8 /= step
                _loss_9 /= step

                total_loss_val = (
                    _loss_1
                    + _loss_2
                    + _loss_3
                    + _loss_4
                    + _loss_5
                    + _loss_6
                    + _loss_7
                    + _loss_8
                    + _loss_9
                )

                writer.add_scalar(
                    "learning rate",
                    optimizer.state_dict()["param_groups"][0]["lr"],
                    global_step=epoch,
                    walltime=None,
                )
                writer.add_scalar("t1_val", _loss_1, global_step=epoch, walltime=None)
                writer.add_scalar("t2_val", _loss_2, global_step=epoch, walltime=None)
                writer.add_scalar("t3_val", _loss_3, global_step=epoch, walltime=None)
                writer.add_scalar("t4_val", _loss_4, global_step=epoch, walltime=None)
                writer.add_scalar("t5_val", _loss_5, global_step=epoch, walltime=None)
                writer.add_scalar("t6_val", _loss_6, global_step=epoch, walltime=None)
                writer.add_scalar("t7_val", _loss_7, global_step=epoch, walltime=None)
                writer.add_scalar("t8_val", _loss_8, global_step=epoch, walltime=None)
                writer.add_scalar("t9_val", _loss_9, global_step=epoch, walltime=None)
                writer.add_scalar(
                    "total_loss_val", total_loss_val, global_step=epoch, walltime=None
                )

                print(
                    f"Val DDF Loss: \n\tloss t1: {_loss_1:.4f}\n\tloss t2: {_loss_2:.4f}\n\tloss t3: {_loss_3:.4f}\n\tloss t4: {_loss_4:.4f}\n\tloss t5: {_loss_5:.4f}\n\tloss t6: {_loss_6:.4f}\n\tloss t7: {_loss_7:.4f}\n\tloss t8: {_loss_8:.4f}\n\tloss t9: {_loss_9:.4f}"
                )
                print(f"Total Loss:{total_loss_val}")
