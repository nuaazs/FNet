from monai.networks.blocks import Warp
import monai
import torch
import torch.nn as nn
from cbamunet import CBAMUNet


class FNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.u01 = CBAMUNet(spatial_dims=3, in_channels=1, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), num_res_units=2, kernel_size=3, up_kernel_size=3, act='PRELU', norm='INSTANCE', dropout=0.0, bias=True, dimensions=None)
        self.warp_layer = Warp()

        # self.u01 =  monai.networks.nets.UNETR(in_channels=1, out_channels=3, img_size=(128,128,64), pos_embed='conv', norm_name='instance')
        self.u01 = monai.networks.nets.AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(32, 64, 128, 256, 32),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.0,
        )
        # self.u03 = monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), kernel_size=3, up_kernel_size=3, dropout=0.0)
        # self.u04 = monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), kernel_size=3, up_kernel_size=3, dropout=0.0)
        # self.u05 = monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), kernel_size=3, up_kernel_size=3, dropout=0.0)
        # self.u06 = monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), kernel_size=3, up_kernel_size=3, dropout=0.0)
        # self.u07 = monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), kernel_size=3, up_kernel_size=3, dropout=0.0)
        # self.u08 = monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), kernel_size=3, up_kernel_size=3, dropout=0.0)
        # self.u09 = monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=1, out_channels=3, channels=(32, 64, 128, 256, 32), strides=(2,2,2,2), kernel_size=3, up_kernel_size=3, dropout=0.0)
        # self.warp_layer = Warp()

    def forward(self, input_images):
        t0 = input_images[:, 0, :, :, :]
        t0 = torch.unsqueeze(t0, 1)
        ddf = self.u01(input_images)
        image = self.warp_layer(t0, ddf)

        return ddf, image


if __name__ == "__main__":
    cbamunet = FNet()
    x = torch.randn(2, 1, 128, 128, 64)
    y = cbamunet(x)
    print(y)
    print(y[0].size())
