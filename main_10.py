from torch.utils.tensorboard import SummaryWriter
import os
import torch
import monai
import numpy as np
import ants
from nets import Net
from dataset_10 import getDataLoader
from torch.nn import MSELoss,L1Loss
from monai.losses import MaskedLoss
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from monai.networks.utils import train_mode,eval_mode,save_state
import logging
import argparse
from utils import show_results,show_error,show_mask
from tqdm import trange


# from corpwechatbot.app import AppMsgSender
# from corpwechatbot import CorpWechatBot

#torch.cuda.empty_cache()

# wechat = AppMsgSender()
# wechat.send_text(content="开始训练 FNet")
torch.backends.cudnn.benchmark = True
    
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
monai.utils.set_determinism(seed=0)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, default="tnet",help='')
parser.add_argument('--batch_size', type=int, default=1,help='')
parser.add_argument('--num_workers', type=int, default=0,help='')
parser.add_argument('--epochs', type=int, default=200,help='')
parser.add_argument('--val_interval', type=int, default=1,help='')
parser.add_argument('--save_interval', type=int, default=10,help='')
parser.add_argument('--istry', action='store_true', default=False,help='')
parser.add_argument('--B2A', action='store_true', default=False,help='')
parser.add_argument('--load', type=int, default=0,help='')
parser.add_argument('--dataset_type',type=str,default="ct",help='')
args = parser.parse_args()

def save_nii(tx,epoch,filename,save_npy_path="./npy"):
    npy = tx.cpu().detach().numpy()[0, 0].transpose((1, 0, 2))
    np.save(os.path.join(save_npy_path,str(epoch)+"_"+filename),npy)
    

for train_index in range(1,10):
    board_path = f"./runs/{args.name}_index_{train_index}"
    os.makedirs(board_path,exist_ok=True)
    os.makedirs(f"./checkpoints/{args.name}_index_{train_index}/",exist_ok=True)
    os.makedirs(f"./npy/{args.name}_index_{train_index}/",exist_ok=True)

    writer = SummaryWriter(board_path)
    train_loader,val_loader = getDataLoader(batch_size=args.batch_size,num_workers=args.num_workers,istry=args.istry,t_index=train_index)
    net = Net(train_index)
    print('# generator parameters:', sum(param.numel() for param in net.parameters()))
    image_loss = L1Loss()
    mask_loss = MaskedLoss(L1Loss)
    optimizer = torch.optim.Adam(net.parameters(), 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, min_lr=1e-6)

    
    if args.load > 0:
        para_in_last_net = torch.load(f"./checkpoints/{args.name}_index_{train_index}/{args.load}.ckpt")
        net.load_state_dict(para_in_last_net)

    with trange(args.load,args.load+args.epochs) as t:
        for epoch in t:
        
            t.set_description(f"Train Index:{train_index} Epoch:{epoch}/{args.load+args.epochs}")     
            

            with train_mode(net):
                epoch_loss=0
                step = 0
                for batch_data in train_loader:
                    step += 1
                    optimizer.zero_grad()
                    input_data = []
                    for tt in range(train_index):
                        input_data.append(batch_data[f"t{tt}_image"])

                    input_data_cuda = torch.cat((input_data),dim=1)
                    ddf,image = net(input_data_cuda)
                    loss_image = image_loss(image,batch_data[f"t{train_index}_image"])
                    loss_ddf = image_loss(ddf,batch_data[f"t{train_index}_ddf"])
                    loss = loss_image + loss_ddf
                    epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                epoch_loss /= step
                writer.add_scalar("learning rate",optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch, walltime=None)
                writer.add_scalar("loss_image",loss_image.item(), global_step=epoch, walltime=None)
                writer.add_scalar("loss_ddf",loss_ddf.item(), global_step=epoch, walltime=None)
                print(f"Train Loss: {epoch_loss:.5f}")

            if (epoch + 1) % args.save_interval == 0:
                save_state(net.state_dict(), f"./checkpoints/{args.name}_index_{train_index}/{epoch}.ckpt")
            
            if (epoch + 1) % args.val_interval == 0:
                with eval_mode(net):
                    val_epoch_loss=0
                    with torch.no_grad():
                        step = 0
                        for batch_data in val_loader:
                            step += 1
                            optimizer.zero_grad()
                            input_data = []
                            for tt in range(train_index):
                                input_data.append(batch_data[f"t{tt}_image"])
                            input_data_cuda = torch.cat((input_data),dim=1)
                            ddf,image = net(input_data_cuda)
                            loss_image = image_loss(image,batch_data[f"t{train_index}_image"])
                            loss_ddf = image_loss(ddf,batch_data[f"t{train_index}_ddf"])
                            pid = batch_data[f"pid"][0]
                            if epoch % args.save_interval == 0:
                                save_nii(image,epoch,f"{pid}_t{tt}_image.npy",save_npy_path="./output_images")
                                save_nii(ddf,epoch,f"{pid}_t{tt}_ddf.npy",save_npy_path="./output_ddfs")
                            loss = loss_image + loss_ddf
                            val_epoch_loss += loss.item()
                        val_epoch_loss /= step
                        writer.add_scalar("val_loss_image",loss_image.item(), global_step=epoch, walltime=None)
                        writer.add_scalar("val_loss_ddf",loss_ddf.item(), global_step=epoch, walltime=None)
                        print(f"Val Loss: {val_epoch_loss:.5f}")
                scheduler.step(val_epoch_loss)
