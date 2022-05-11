import torch
import numpy as np
from matplotlib import pyplot as plt

    
def show_results(writer,results,batch_data,step,suffix):
    figure = plt.figure(figsize=(10,5),dpi=200,frameon=False)
    for i in range(1,6):
        ax=plt.subplot(1,5,i)
        img_tensor = torch.cat((results[i-1],batch_data[f"t{i}_image"].cuda()), dim=-1)[0,0,:,:,:]
        npy = img_tensor.cpu().detach().numpy()
        #np.save("./test.npy",npy)
        ax.imshow(npy.transpose(2,0,1)[::-1,:,50],cmap='gray')
        ax.set_title(f"T{i}")
        ax.set_xticks([])
        ax.set_yticks([])

    #plt.show()
    writer.add_figure(f"{suffix}",figure,global_step=step)

def show_error(writer,results,batch_data,step,suffix):
    figure = plt.figure(figsize=(10,5),dpi=200,frameon=False)
    for i in range(1,6):
        ax=plt.subplot(1,5,i)
        img_tensor = results[i-1][0,0,:,:,:] - batch_data[f"t{i}_image"].cuda()[0,0,:,:,:]
        npy = img_tensor.cpu().detach().numpy()
        #np.save("./test.npy",npy)
        ax.imshow(npy.transpose(2,0,1)[::-1,:,50],cmap='gray')
        ax.set_title(f"T{i}")
        ax.set_xticks([])
        ax.set_yticks([])
    #plt.show()
    writer.add_figure(f"{suffix}_Err",figure,global_step=step)

def show_mask(writer,results,batch_data,step,suffix):
    figure = plt.figure(figsize=(10,5),dpi=200,frameon=False)
    for i in range(1,6):
        ax=plt.subplot(1,5,i)
        img_tensor = batch_data[f"t{i}_mask"].cuda()[0,0,:,:,:]
        npy = img_tensor.cpu().detach().numpy()
        #np.save("./test.npy",npy)
        ax.imshow(npy.transpose(2,0,1)[::-1,:,50],cmap='gray')
        ax.set_title(f"T{i}")
        ax.set_xticks([])
        ax.set_yticks([])
    #plt.show()
    writer.add_figure(f"{suffix}_Mask",figure,global_step=step)