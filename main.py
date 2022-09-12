from torch.utils.tensorboard import SummaryWriter
import os
import torch
import monai
import numpy as np
from backbone.nets import FNet
from datasets.dataset import getDataLoader
from torch.nn import MSELoss,L1Loss
from monai.losses import MaskedLoss
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from monai.networks.utils import train_mode,eval_mode,save_state
import logging
import argparse
from utils.save import save_nii
import random
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True   

seed = 123456
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
monai.utils.set_determinism(seed=seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, default="fnet0904",help='')
parser.add_argument('--ddf_dir', type=str, default="/mnt/zhaosheng/FNet/data/ddfs",help='')
parser.add_argument('--img_dir', type=str, default="/mnt/zhaosheng/4dct/resampled",help='')
parser.add_argument('--ddf_prefix', type=str, default="49",help='')
parser.add_argument('--batch_size', type=int, default=1,help='')
parser.add_argument('--num_workers', type=int, default=0,help='')
parser.add_argument('--epochs', type=int, default=200,help='')
parser.add_argument('--val_interval', type=int, default=1,help='')
parser.add_argument('--save_interval', type=int, default=20,help='')
parser.add_argument('--save_npy_interval', type=int, default=20,help='')
parser.add_argument('--istry', action='store_true', default=False,help='')
parser.add_argument('--B2A', action='store_true', default=False,help='')
parser.add_argument('--load', type=int, default=0,help='')
parser.add_argument('--dataset_type',type=str,default="ct",help='')
args = parser.parse_args()


os.makedirs(f"./runs/{args.name}",exist_ok=True)
os.makedirs(f"./checkpoints/{args.name}/",exist_ok=True)
os.makedirs(f"./results/{args.name}/",exist_ok=True)

writer = SummaryWriter(f"./runs/{args.name}")
train_loader,val_loader = getDataLoader(batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        istry=args.istry,
                                        ddf_prefix=args.ddf_prefix,
                                        ddf_dir=args.ddf_dir,
                                        img_dir = args.img_dir)
net = FNet()
print('# generator parameters:', sum(param.numel() for param in net.parameters()))
net = net.cuda()
gpus = [0,1]
net = torch.nn.DataParallel(net, device_ids=gpus, output_device=gpus[0])

image_loss = L1Loss()
mask_loss = MaskedLoss(L1Loss)
optimizer = torch.optim.Adam(net.parameters(), 1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, min_lr=1e-6)

if args.load > 0:
    para_in_last_net = torch.load(f"./checkpoints/{args.name}/{args.load}.ckpt")
    net.load_state_dict(para_in_last_net)

for epoch in range(args.load,args.load+args.epochs):
    print(f"epoch {epoch + 1}/{args.load+args.epochs}")
    with train_mode(net):
        
        image_loss_t1 = 0 
        image_loss_t2 = 0 
        image_loss_t3 = 0 
        image_loss_t4 = 0 
        image_loss_t5 = 0 
        image_loss_t6 = 0 
        image_loss_t7 = 0 
        image_loss_t8 = 0 
        image_loss_t9 = 0 
        ddf_loss_1  = 0
        ddf_loss_2  = 0
        ddf_loss_3  = 0
        ddf_loss_4  = 0
        ddf_loss_5  = 0
        ddf_loss_6  = 0
        ddf_loss_7  = 0
        ddf_loss_8  = 0
        ddf_loss_9  = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            optimizer.zero_grad()
            t0_image = batch_data["t0_image"].cuda()
            results = net(t0_image)
            ddf1,ddf2,ddf3,ddf4,ddf5,ddf6,ddf7,ddf8,ddf9,image1,image2,image3,image4,image5,image6,image7,image8,image9 = results

            loss_image_t1 = image_loss(image1,batch_data["t1_image"].cuda()) * 100
            loss_image_t2 = image_loss(image2,batch_data["t2_image"].cuda()) * 100
            loss_image_t3 = image_loss(image3,batch_data["t3_image"].cuda()) * 100
            loss_image_t4 = image_loss(image4,batch_data["t4_image"].cuda()) * 100
            loss_image_t5 = image_loss(image5,batch_data["t5_image"].cuda()) * 100
            loss_image_t6 = image_loss(image6,batch_data["t6_image"].cuda()) * 100
            loss_image_t7 = image_loss(image7,batch_data["t7_image"].cuda()) * 100
            loss_image_t8 = image_loss(image8,batch_data["t8_image"].cuda()) * 100
            loss_image_t9 = image_loss(image9,batch_data["t9_image"].cuda()) * 100

            loss_ddf_1 = image_loss(ddf1,batch_data["ddf1_image"].cuda())
            loss_ddf_2 = image_loss(ddf2,batch_data["ddf2_image"].cuda())
            loss_ddf_3 = image_loss(ddf3,batch_data["ddf3_image"].cuda())
            loss_ddf_4 = image_loss(ddf4,batch_data["ddf4_image"].cuda())
            loss_ddf_5 = image_loss(ddf5,batch_data["ddf5_image"].cuda())
            loss_ddf_6 = image_loss(ddf6,batch_data["ddf6_image"].cuda())
            loss_ddf_7 = image_loss(ddf7,batch_data["ddf7_image"].cuda())
            loss_ddf_8 = image_loss(ddf8,batch_data["ddf8_image"].cuda())
            loss_ddf_9 = image_loss(ddf9,batch_data["ddf9_image"].cuda())

            image_loss_t1 += loss_image_t1.item()
            image_loss_t2 += loss_image_t2.item()
            image_loss_t3 += loss_image_t3.item()
            image_loss_t4 += loss_image_t4.item()
            image_loss_t5 += loss_image_t5.item()
            image_loss_t6 += loss_image_t6.item()
            image_loss_t7 += loss_image_t7.item()
            image_loss_t8 += loss_image_t8.item()
            image_loss_t9 += loss_image_t9.item()

            ddf_loss_1 += loss_ddf_1.item()
            ddf_loss_2 += loss_ddf_2.item()
            ddf_loss_3 += loss_ddf_3.item()
            ddf_loss_4 += loss_ddf_4.item()
            ddf_loss_5 += loss_ddf_5.item()
            ddf_loss_6 += loss_ddf_6.item()
            ddf_loss_7 += loss_ddf_7.item()
            ddf_loss_8 += loss_ddf_8.item()
            ddf_loss_9 += loss_ddf_9.item()
            print(loss_image_t9.item(),loss_ddf_9.item())
            loss = loss_image_t1+loss_image_t2+loss_image_t3+loss_image_t4+loss_image_t5+loss_image_t6+loss_image_t7+loss_image_t8+loss_image_t9
            loss += loss_ddf_1 + loss_ddf_2 + loss_ddf_3 + loss_ddf_4 + loss_ddf_5 + loss_ddf_6 + loss_ddf_7 + loss_ddf_8 + loss_ddf_9
            loss.backward()
            optimizer.step()
        image_loss_t1 /= step
        image_loss_t2 /= step
        image_loss_t3 /= step
        image_loss_t4 /= step
        image_loss_t5 /= step
        image_loss_t6 /= step
        image_loss_t7 /= step
        image_loss_t8 /= step
        image_loss_t9 /= step

        ddf_loss_1 /= step
        ddf_loss_2 /= step
        ddf_loss_3 /= step
        ddf_loss_4 /= step
        ddf_loss_5 /= step
        ddf_loss_6 /= step
        ddf_loss_7 /= step
        ddf_loss_8 /= step
        ddf_loss_9 /= step

        total_loss_train = image_loss_t1 + image_loss_t2 + image_loss_t3 + image_loss_t4 + image_loss_t5 \
             + image_loss_t6 + image_loss_t7 + image_loss_t8 + image_loss_t9 \
             + ddf_loss_1 + ddf_loss_2 + ddf_loss_3 + ddf_loss_4 + ddf_loss_5 \
             + ddf_loss_6 + ddf_loss_7 + ddf_loss_8 + ddf_loss_9

        writer.add_scalar("learning rate",optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch, walltime=None)
        writer.add_scalar("t1",image_loss_t1, global_step=epoch, walltime=None)
        writer.add_scalar("t2",image_loss_t2, global_step=epoch, walltime=None)
        writer.add_scalar("t3",image_loss_t3, global_step=epoch, walltime=None)
        writer.add_scalar("t4",image_loss_t4, global_step=epoch, walltime=None)
        writer.add_scalar("t5",image_loss_t5, global_step=epoch, walltime=None)
        writer.add_scalar("t6",image_loss_t6, global_step=epoch, walltime=None)
        writer.add_scalar("t7",image_loss_t7, global_step=epoch, walltime=None)
        writer.add_scalar("t8",image_loss_t8, global_step=epoch, walltime=None)
        writer.add_scalar("t9",image_loss_t9, global_step=epoch, walltime=None)

        writer.add_scalar("ddf1",ddf_loss_1, global_step=epoch, walltime=None)
        writer.add_scalar("ddf2",ddf_loss_2, global_step=epoch, walltime=None)
        writer.add_scalar("ddf3",ddf_loss_3, global_step=epoch, walltime=None)
        writer.add_scalar("ddf4",ddf_loss_4, global_step=epoch, walltime=None)
        writer.add_scalar("ddf5",ddf_loss_5, global_step=epoch, walltime=None)
        writer.add_scalar("ddf6",ddf_loss_6, global_step=epoch, walltime=None)
        writer.add_scalar("ddf7",ddf_loss_7, global_step=epoch, walltime=None)
        writer.add_scalar("ddf8",ddf_loss_8, global_step=epoch, walltime=None)
        writer.add_scalar("ddf9",ddf_loss_9, global_step=epoch, walltime=None)
        print(f"Train Loss: {total_loss_train:.5f}\nTrain average -> \n\tloss t1: {image_loss_t1:.4f}\n\tloss t2: {image_loss_t2:.4f}\n\tloss t3: {image_loss_t3:.4f}\n\tloss t4: {image_loss_t4:.4f}\n\tloss t5: {image_loss_t5:.4f}\n\tloss t6: {image_loss_t6:.4f}\n\tloss t7: {image_loss_t7:.4f}\n\tloss t8: {image_loss_t8:.4f}\n\tloss t9: {image_loss_t9:.4f}")
        print(f"DDF Loss: \n\tloss t1: {ddf_loss_1:.4f}\n\tloss t2: {ddf_loss_2:.4f}\n\tloss t3: {ddf_loss_3:.4f}\n\tloss t4: {ddf_loss_4:.4f}\n\tloss t5: {ddf_loss_5:.4f}\n\tloss t6: {ddf_loss_6:.4f}\n\tloss t7: {ddf_loss_7:.4f}\n\tloss t8: {ddf_loss_8:.4f}\n\tloss t9: {ddf_loss_9:.4f}")

    if (epoch + 1) % args.save_interval == 0:
        save_state(net.state_dict(), f"./checkpoints/{args.name}/{epoch}.ckpt")
        
    if (epoch + 1) % args.val_interval == 0:
        with eval_mode(net):
            epoch_loss_t1,epoch_loss_t2,epoch_loss_t3,epoch_loss_t4,epoch_loss_t5= 0,0,0,0,0
            with torch.no_grad():
                step = 0
            for batch_data in val_loader:
                step += 1
                optimizer.zero_grad()
                t0_image = batch_data["t0_image"].cuda()
                results = net(t0_image)
                ddf1,ddf2,ddf3,ddf4,ddf5,ddf6,ddf7,ddf8,ddf9,image1,image2,image3,image4,image5,image6,image7,image8,image9 = results
                if (epoch + 1) % args.save_npy_interval == 0:
                    fake_ddf = torch.cat([ddf1,ddf2,ddf3,ddf4,ddf5,ddf6,ddf7,ddf8,ddf9],1)
                    real_ddf = torch.cat([batch_data["ddf1_image"],batch_data["ddf2_image"],batch_data["ddf3_image"],batch_data["ddf4_image"],batch_data["ddf5_image"],batch_data["ddf6_image"],batch_data["ddf7_image"],batch_data["ddf8_image"],batch_data["ddf9_image"]],1)
                    fake_img = torch.cat([image1,image2,image3,image4,image5,image6,image7,image8,image9],1)
                    real_img = torch.cat([batch_data["t1_image"],batch_data["t2_image"],batch_data["t3_image"],batch_data["t4_image"],batch_data["t5_image"],batch_data["t6_image"],batch_data["t7_image"],batch_data["t8_image"],batch_data["t9_image"]],1)
                    save_nii(tensor=fake_ddf,filename=str(epoch)+"_"+batch_data["pid"][0]+"_ddf_fake",save_npy_path=f"./results/{args.name}/")
                    save_nii(tensor=real_ddf,filename=str(epoch)+"_"+batch_data["pid"][0]+"_ddf_real",save_npy_path=f"./results/{args.name}/")
                    save_nii(tensor=fake_img,filename=str(epoch)+"_"+batch_data["pid"][0]+"_image_fake",save_npy_path=f"./results/{args.name}/")
                    save_nii(tensor=real_img,filename=str(epoch)+"_"+batch_data["pid"][0]+"_image_real",save_npy_path=f"./results/{args.name}/")
                

                loss_image_t1 = image_loss(image1,batch_data["t1_image"].cuda()) * 100
                loss_image_t2 = image_loss(image2,batch_data["t2_image"].cuda()) * 100
                loss_image_t3 = image_loss(image3,batch_data["t3_image"].cuda()) * 100
                loss_image_t4 = image_loss(image4,batch_data["t4_image"].cuda()) * 100
                loss_image_t5 = image_loss(image5,batch_data["t5_image"].cuda()) * 100
                loss_image_t6 = image_loss(image6,batch_data["t6_image"].cuda()) * 100
                loss_image_t7 = image_loss(image7,batch_data["t7_image"].cuda()) * 100
                loss_image_t8 = image_loss(image8,batch_data["t8_image"].cuda()) * 100
                loss_image_t9 = image_loss(image9,batch_data["t9_image"].cuda()) * 100

                loss_ddf_1 = image_loss(ddf1,batch_data["ddf1_image"].cuda())
                loss_ddf_2 = image_loss(ddf2,batch_data["ddf2_image"].cuda())
                loss_ddf_3 = image_loss(ddf3,batch_data["ddf3_image"].cuda())
                loss_ddf_4 = image_loss(ddf4,batch_data["ddf4_image"].cuda())
                loss_ddf_5 = image_loss(ddf5,batch_data["ddf5_image"].cuda())
                loss_ddf_6 = image_loss(ddf6,batch_data["ddf6_image"].cuda())
                loss_ddf_7 = image_loss(ddf7,batch_data["ddf7_image"].cuda())
                loss_ddf_8 = image_loss(ddf8,batch_data["ddf8_image"].cuda())
                loss_ddf_9 = image_loss(ddf9,batch_data["ddf9_image"].cuda())

                image_loss_t1 += loss_image_t1.item()
                image_loss_t2 += loss_image_t2.item()
                image_loss_t3 += loss_image_t3.item()
                image_loss_t4 += loss_image_t4.item()
                image_loss_t5 += loss_image_t5.item()
                image_loss_t6 += loss_image_t6.item()
                image_loss_t7 += loss_image_t7.item()
                image_loss_t8 += loss_image_t8.item()
                image_loss_t9 += loss_image_t9.item()

                ddf_loss_1 += loss_ddf_1.item()
                ddf_loss_2 += loss_ddf_2.item()
                ddf_loss_3 += loss_ddf_3.item()
                ddf_loss_4 += loss_ddf_4.item()
                ddf_loss_5 += loss_ddf_5.item()
                ddf_loss_6 += loss_ddf_6.item()
                ddf_loss_7 += loss_ddf_7.item()
                ddf_loss_8 += loss_ddf_8.item()
                ddf_loss_9 += loss_ddf_9.item()
                total_loss_val = image_loss_t1 + image_loss_t2 + image_loss_t3 + image_loss_t4 + image_loss_t5 \
                                    + image_loss_t6 + image_loss_t7 + image_loss_t8 + image_loss_t9 \
                                    + ddf_loss_1 + ddf_loss_2 + ddf_loss_3 + ddf_loss_4 + ddf_loss_5 \
                                    + ddf_loss_6 + ddf_loss_7 + ddf_loss_8 + ddf_loss_9
            

            writer.add_scalar("val_t1",image_loss_t1, global_step=epoch, walltime=None)
            writer.add_scalar("val_t2",image_loss_t2, global_step=epoch, walltime=None)
            writer.add_scalar("val_t3",image_loss_t3, global_step=epoch, walltime=None)
            writer.add_scalar("val_t4",image_loss_t4, global_step=epoch, walltime=None)
            writer.add_scalar("val_t5",image_loss_t5, global_step=epoch, walltime=None)
            writer.add_scalar("val_t6",image_loss_t6, global_step=epoch, walltime=None)
            writer.add_scalar("val_t7",image_loss_t7, global_step=epoch, walltime=None)
            writer.add_scalar("val_t8",image_loss_t8, global_step=epoch, walltime=None)
            writer.add_scalar("val_t9",image_loss_t9, global_step=epoch, walltime=None)

            writer.add_scalar("val_ddf1",ddf_loss_1, global_step=epoch, walltime=None)
            writer.add_scalar("val_ddf2",ddf_loss_2, global_step=epoch, walltime=None)
            writer.add_scalar("val_ddf3",ddf_loss_3, global_step=epoch, walltime=None)
            writer.add_scalar("val_ddf4",ddf_loss_4, global_step=epoch, walltime=None)
            writer.add_scalar("val_ddf5",ddf_loss_5, global_step=epoch, walltime=None)
            writer.add_scalar("val_ddf6",ddf_loss_6, global_step=epoch, walltime=None)
            writer.add_scalar("val_ddf7",ddf_loss_7, global_step=epoch, walltime=None)
            writer.add_scalar("val_ddf8",ddf_loss_8, global_step=epoch, walltime=None)
            writer.add_scalar("val_ddf9",ddf_loss_9, global_step=epoch, walltime=None)
            print(f"Val Loss: {total_loss_val:.5f}\nTrain average -> \n\tloss t1: {image_loss_t1:.4f}\n\tloss t2: {image_loss_t2:.4f}\n\tloss t3: {image_loss_t3:.4f}\n\tloss t4: {image_loss_t4:.4f}\n\tloss t5: {image_loss_t5:.4f}\n\tloss t6: {image_loss_t6:.4f}\n\tloss t7: {image_loss_t7:.4f}\n\tloss t8: {image_loss_t8:.4f}\n\tloss t9: {image_loss_t9:.4f}")
            print(f"Val DDF Loss: \n\tloss t1: {ddf_loss_1:.4f}\n\tloss t2: {ddf_loss_2:.4f}\n\tloss t3: {ddf_loss_3:.4f}\n\tloss t4: {ddf_loss_4:.4f}\n\tloss t5: {ddf_loss_5:.4f}\n\tloss t6: {ddf_loss_6:.4f}\n\tloss t7: {ddf_loss_7:.4f}\n\tloss t8: {ddf_loss_8:.4f}\n\tloss t9: {ddf_loss_9:.4f}")