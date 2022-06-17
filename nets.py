from monai.networks.blocks import Warp
import monai
import torch
import torch.nn as nn
from cbamunet import CBAMUNet

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.u01 = CBAMUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        # self.u02 = CBAMUNet(spatial_dims=3, in_channels=2, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        # self.u03 = CBAMUNet(spatial_dims=3, in_channels=3, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        # self.u04 = CBAMUNet(spatial_dims=3, in_channels=4, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        # self.u05 = CBAMUNet(spatial_dims=3, in_channels=5, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        self.u01 = CBAMUNet(spatial_dims=3, in_channels=1, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), num_res_units=2, kernel_size=3, up_kernel_size=3, act='PRELU', norm='INSTANCE', dropout=0.0, bias=True, dimensions=None)
        self.u02 = CBAMUNet(spatial_dims=3, in_channels=2, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), num_res_units=2, kernel_size=3, up_kernel_size=3, act='PRELU', norm='INSTANCE', dropout=0.0, bias=True, dimensions=None)
        self.u03 = CBAMUNet(spatial_dims=3, in_channels=3, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), num_res_units=2, kernel_size=3, up_kernel_size=3, act='PRELU', norm='INSTANCE', dropout=0.0, bias=True, dimensions=None)
        self.u04 = CBAMUNet(spatial_dims=3, in_channels=4, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), num_res_units=2, kernel_size=3, up_kernel_size=3, act='PRELU', norm='INSTANCE', dropout=0.0, bias=True, dimensions=None)
        self.u05 = CBAMUNet(spatial_dims=3, in_channels=5, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), num_res_units=2, kernel_size=3, up_kernel_size=3, act='PRELU', norm='INSTANCE', dropout=0.0, bias=True, dimensions=None)
        
        
        self.warp_layer = Warp()
    def forward(self, t0):
        ddf01 = self.u01(t0)
        t1 = self.warp_layer(t0, ddf01)
        ddf02 = self.u02(torch.cat((t0,t1), dim=1))
        t2 = self.warp_layer(t0, ddf02)
        ddf03 = self.u03(torch.cat((t0,t1,t2), dim=1))
        t3 = self.warp_layer(t0, ddf03)
        ddf04 = self.u04(torch.cat((t0,t1,t2,t3), dim=1))
        t4 = self.warp_layer(t0, ddf04)
        ddf05 = self.u05(torch.cat((t0,t1,t2,t3,t4), dim=1))
        t5 = self.warp_layer(t0, ddf05)
        return [ddf01,ddf02,ddf03,ddf04,ddf05,t1,t2,t3,t4,t5]

if __name__ == "__main__":
    cbamunet = Net()
    x = torch.randn(2,1,128,128,64)
    y = cbamunet(x)
    print(y)
    print(y[0].size())
