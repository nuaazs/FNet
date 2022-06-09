from monai.networks.blocks import Warp
import monai
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.u01 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        self.u02 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=2, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        self.u03 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=3, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        self.u04 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=4, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        self.u05 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=5, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        #self.u56 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        #self.u67 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        #self.u78 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        #self.u89 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        #self.u90 = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=3, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv', dimensions=None)
        self.warp_layer = Warp()
    def forward(self, t0):
        ddf01 = self.u01(t0)
        t1 = self.warp_layer(t0, ddf01)
        ddf02 = self.u02(torch.cat((t0,t1), dim=1))
        t2 = self.warp_layer(t0, t1)
        ddf03 = self.u03(torch.cat((t0,t1,t2), dim=1))
        t3 = self.warp_layer(t0, t2)
        ddf04 = self.u04(torch.cat((t0,t1,t2,t3), dim=1))
        t4 = self.warp_layer(t0, t3)
        ddf05 = self.u05(torch.cat((t0,t1,t2,t3,t4), dim=1))
        
        return [ddf01,ddf02,ddf03,ddf04,ddf05,t1,t2,t3,t4,t5]
