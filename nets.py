from monai.networks.blocks import Warp
import monai
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.u01 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        self.u12 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=2, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        self.u23 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=3, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        self.u34 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=4, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        self.u45 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=5, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        #self.u56 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        #self.u67 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        #self.u78 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        #self.u89 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        #self.u90 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        self.warp_layer = Warp()
    def forward(self, t0):
        ddf01 = self.u01(t0)
        t1 = self.warp_layer(t0, ddf01)
        ddf02 = self.u12(torch.cat((t0,t1), dim=1))
        t2 = self.warp_layer(t0, ddf02)
        ddf03 = self.u23(torch.cat((t0,t1,t2), dim=1))
        t3 = self.warp_layer(t0, ddf03)
        ddf04 = self.u34(torch.cat((t0,t1,t2,t3), dim=1))
        t4 = self.warp_layer(t0, ddf04)
        ddf05 = self.u45(torch.cat((t0,t1,t2,t3,t4), dim=1))
        t5 = self.warp_layer(t0, ddf05)
        # ddf56 = self.u56(t5)
        # t6 = self.warp_layer(t5, ddf56)
        # ddf67 = self.u67(t6)
        # t7 = self.warp_layer(t6, ddf67)
        # ddf78 = self.u78(t7)
        # t8 = self.warp_layer(t7, ddf78)
        # ddf89 = self.u89(t8)
        # t9 = self.warp_layer(t8, ddf89)
        # ddf90 = self.u90(t9)
        # ft0 = self.warp_layer(t9, ddf90)
        #print(f"ddf01 shape:{ddf01.shape}")
        #print(f"ddf01 min:{ddf01.min()}")
        #print(f"ddf01 max:{ddf01.max()}")
        return [ddf01,ddf02,ddf03,ddf04,ddf05,t1,t2,t3,t4,t5]#,t6,t7,t8,t9,ft0
