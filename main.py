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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from monai.networks.utils import train_mode,eval_mode,save_state
import logging
import argparse
from utils import show_results,show_error,show_mask
# from corpwechatbot.app import AppMsgSender
# from corpwechatbot import CorpWechatBot
torch.cuda.empty_cache()
# wechat = AppMsgSender()
# wechat.send_text(content="开始训练 FNet")
torch.backends.cudnn.benchmark = True
    
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
monai.utils.set_determinism(seed=0)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, default="tnet",help='')
parser.add_argument('--batch_size', type=int, default=2,help='')
parser.add_argument('--num_workers', type=int, default=0,help='')
parser.add_argument('--epochs', type=int, default=200,help='')
parser.add_argument('--val_interval', type=int, default=1,help='')
parser.add_argument('--save_interval', type=int, default=5,help='')
parser.add_argument('--istry', action='store_true', default=False,help='')
parser.add_argument('--B2A', action='store_true', default=False,help='')
parser.add_argument('--load', type=int, default=0,help='')
parser.add_argument('--dataset_type',type=str,default="ct",help='')
args = parser.parse_args()



board_path = f"./runs/{args.name}"
os.makedirs(board_path,exist_ok=True)
os.makedirs(f"./checkpoints/{args.name}/",exist_ok=True)
os.makedirs(f"./npy/{args.name}/",exist_ok=True)
writer = SummaryWriter(board_path)
train_loader,val_loader = getDataLoader(batch_size=args.batch_size,num_workers=args.num_workers,istry=args.istry)
net = Net()
print('# generator parameters:', sum(param.numel() for param in net.parameters()))
net = net.cuda()
gpus = [0,1]
net = torch.nn.DataParallel(net, device_ids=gpus, output_device=gpus[0])


image_loss = L1Loss()
mask_loss = MaskedLoss(L1Loss)
optimizer = torch.optim.Adam(net.parameters(), 1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, min_lr=1e-6)

def save_nii(tx,epoch,filename,save_npy_path="./npy"):
    npy = tx.cpu().detach().numpy()[0, 0].transpose((1, 0, 2))
    np.save(os.path.join(save_npy_path,str(epoch)+"_"+filename),npy)

if args.load > 0:
    para_in_last_net = torch.load(f"./checkpoints/{args.name}/{args.load}.ckpt")
    net.load_state_dict(para_in_last_net)


for epoch in range(args.load,args.load+args.epochs):
    print(f"epoch {epoch + 1}/{args.load+args.epochs}")
    with train_mode(net):
        epoch_loss_t1,epoch_loss_t2,epoch_loss_t3,epoch_loss_t4,epoch_loss_t5= 0,0,0,0,0
        step = 0
        for batch_data in train_loader:
            step += 1
            optimizer.zero_grad()
            t0_image = batch_data["t0_image"].cuda()
            results = net(t0_image)
            ddf01,ddf02,ddf03,ddf04,ddf05,t1,t2,t3,t4,t5 = results

            if not args.B2A:
                loss_image_t1 = image_loss(t1,batch_data["t1_image"].cuda())
                loss_image_t2 = image_loss(t2,batch_data["t2_image"].cuda())
                loss_image_t3 = image_loss(t3,batch_data["t3_image"].cuda())
                loss_image_t4 = image_loss(t4,batch_data["t4_image"].cuda())
                loss_image_t5 = image_loss(t5,batch_data["t5_image"].cuda())
            else:
                loss_image_t1 = image_loss(t1,batch_data["t9_image"].cuda())
                loss_image_t2 = image_loss(t2,batch_data["t8_image"].cuda())
                loss_image_t3 = image_loss(t3,batch_data["t7_image"].cuda())
                loss_image_t4 = image_loss(t4,batch_data["t6_image"].cuda())
                loss_image_t5 = image_loss(t5,batch_data["t5_image"].cuda())

            epoch_loss_t1 += loss_image_t1.item()
            epoch_loss_t2 += loss_image_t2.item()
            epoch_loss_t3 += loss_image_t3.item()
            epoch_loss_t4 += loss_image_t4.item()
            epoch_loss_t5 += loss_image_t5.item()
            loss = loss_image_t1+loss_image_t2+loss_image_t3+loss_image_t4+loss_image_t5
            loss.backward()
            optimizer.step()
        epoch_loss_t1 /= step
        epoch_loss_t2 /= step
        epoch_loss_t3 /= step
        epoch_loss_t4 /= step
        epoch_loss_t5 /= step
        total_loss_train = epoch_loss_t1+ epoch_loss_t2+ epoch_loss_t3+ epoch_loss_t4+ epoch_loss_t5
        writer.add_scalar("learning rate",optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch, walltime=None)
        
        writer.add_scalar("t1",epoch_loss_t1, global_step=epoch, walltime=None)
        writer.add_scalar("t2",epoch_loss_t2, global_step=epoch, walltime=None)
        writer.add_scalar("t3",epoch_loss_t3, global_step=epoch, walltime=None)
        writer.add_scalar("t4",epoch_loss_t4, global_step=epoch, walltime=None)
        writer.add_scalar("t5",epoch_loss_t5, global_step=epoch, walltime=None)
        print(f"Train Loss: {total_loss_train:.5f}\nVal average -> \n\tloss t1: {epoch_loss_t1:.4f}\n\tloss t2: {epoch_loss_t2:.4f}\n\tloss t3: {epoch_loss_t3:.4f}\n\tloss t4: {epoch_loss_t4:.4f}\n\tloss t5: {epoch_loss_t5:.4f}")

    if (epoch + 1) % args.save_interval == 0:
        save_state(net.state_dict(), f"./checkpoints/{args.name}/{epoch}.ckpt")
    
    if (epoch + 1) % args.val_interval == 0:
        with eval_mode(net):
            epoch_loss_t1,epoch_loss_t2,epoch_loss_t3,epoch_loss_t4,epoch_loss_t5= 0,0,0,0,0
            with torch.no_grad():
                step = 0
                for index,batch_data in enumerate(val_loader):
                    step += 1
                    t0_image = batch_data["t0_image"].cuda()
                    results = net(t0_image)
                    ddf01,ddf02,ddf03,ddf04,ddf05,t1,t2,t3,t4,t5 = results

                    if not args.B2A:
                        loss_image_t1 = image_loss(t1,batch_data["t1_image"].cuda())
                        loss_image_t2 = image_loss(t2,batch_data["t2_image"].cuda())
                        loss_image_t3 = image_loss(t3,batch_data["t3_image"].cuda())
                        loss_image_t4 = image_loss(t4,batch_data["t4_image"].cuda())
                        loss_image_t5 = image_loss(t5,batch_data["t5_image"].cuda())
                    else:
                        loss_image_t1 = image_loss(t1,batch_data["t9_image"].cuda())
                        loss_image_t2 = image_loss(t2,batch_data["t8_image"].cuda())
                        loss_image_t3 = image_loss(t3,batch_data["t7_image"].cuda())
                        loss_image_t4 = image_loss(t4,batch_data["t6_image"].cuda())
                        loss_image_t5 = image_loss(t5,batch_data["t5_image"].cuda())
                
                    epoch_loss_t1 += loss_image_t1.item()
                    epoch_loss_t2 += loss_image_t2.item()
                    epoch_loss_t3 += loss_image_t3.item()
                    epoch_loss_t4 += loss_image_t4.item()
                    epoch_loss_t5 += loss_image_t5.item()
                epoch_loss_t1 /= step
                epoch_loss_t2 /= step
                epoch_loss_t3 /= step
                epoch_loss_t4 /= step
                epoch_loss_t5 /= step
                total_loss_val = epoch_loss_t1+ epoch_loss_t2+ epoch_loss_t3+ epoch_loss_t4+ epoch_loss_t5
                writer.add_scalar("val_t1",epoch_loss_t1, global_step=epoch, walltime=None)
                writer.add_scalar("val_t2",epoch_loss_t2, global_step=epoch, walltime=None)
                writer.add_scalar("val_t3",epoch_loss_t3, global_step=epoch, walltime=None)
                writer.add_scalar("val_t4",epoch_loss_t4, global_step=epoch, walltime=None)
                writer.add_scalar("val_t5",epoch_loss_t5, global_step=epoch, walltime=None)
                print(f"Val Loss: {total_loss_val:.5f}\nVal average -> \n\tloss t1: {epoch_loss_t1:.4f}\n\tloss t2: {epoch_loss_t2:.4f}\n\tloss t3: {epoch_loss_t3:.4f}\n\tloss t4: {epoch_loss_t4:.4f}\n\tloss t5: {epoch_loss_t5:.4f}")
        scheduler.step(total_loss_val)

# wechat.send_text(content="训练完成！")
