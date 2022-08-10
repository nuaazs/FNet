from monai.networks.blocks import Warp
import monai
import torch
import torch.nn as nn
from cbamunet import CBAMUNet

class Net(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.u01 = CBAMUNet(spatial_dims=3, in_channels=in_channels, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), num_res_units=2, kernel_size=3, up_kernel_size=3, act='PRELU', norm='INSTANCE', dropout=0.0, bias=True, dimensions=None)
        self.warp_layer = Warp()
    def forward(self,input_images):
        t0 =input_images[:,0,:,:,:]
        t0 = torch.unsqueeze(t0,1)
        ddf = self.u01(input_images)
        image = self.warp_layer(t0, ddf)
        return ddf,image

if __name__ == "__main__":
    cbamunet = Net()
    x = torch.randn(2,1,128,128,64)
    y = cbamunet(x)
    print(y)
    print(y[0].size())