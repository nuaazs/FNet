from monai.networks.blocks import Warp
import monai
import torch
import torch.nn as nn
from cbamunet import CBAMUNet


class FNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.u01 = CBAMUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(32, 64, 128, 256, 32),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
            bias=True,
            dimensions=None,
        )
        self.u02 = CBAMUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            channels=(32, 64, 128, 256, 32),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
            bias=True,
            dimensions=None,
        )
        self.u03 = CBAMUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=3,
            channels=(32, 64, 128, 256, 32),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
            bias=True,
            dimensions=None,
        )
        self.u04 = CBAMUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(32, 64, 128, 256, 32),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
            bias=True,
            dimensions=None,
        )
        self.u05 = CBAMUNet(
            spatial_dims=3,
            in_channels=5,
            out_channels=3,
            channels=(32, 64, 128, 256, 32),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
            bias=True,
            dimensions=None,
        )
        self.u06 = CBAMUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(32, 64, 128, 256, 32),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
            bias=True,
            dimensions=None,
        )
        self.u07 = CBAMUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=3,
            channels=(32, 64, 128, 256, 32),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
            bias=True,
            dimensions=None,
        )
        self.u08 = CBAMUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            channels=(32, 64, 128, 256, 32),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
            bias=True,
            dimensions=None,
        )
        self.u09 = CBAMUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(32, 64, 128, 256, 32),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
            bias=True,
            dimensions=None,
        )
        self.warp_layer = Warp()

    def forward(self, input_images):
        t0 = input_images  # [:, 0, :, :, :]
        # t0 = torch.unsqueeze(t0, 1)
        # print(f"t0.shape: {t0.shape}")
        # print(/)
        ddf1 = self.u01(input_images)
        # print(ddf1.shape)
        image1 = self.warp_layer(t0, ddf1)
        # print(f"image1.shape: {image1.shape}")
        ddf2 = self.u02(torch.cat([t0, image1], 1))  # + ddf1
        image2 = self.warp_layer(t0, ddf2)
        ddf3 = self.u03(torch.cat([t0, image1, image2], 1))  # + ddf2
        image3 = self.warp_layer(t0, ddf3)
        ddf4 = self.u04(torch.cat([t0, image1, image2, image3], 1))  # + ddf3
        image4 = self.warp_layer(t0, ddf4)

        ddf9 = self.u09(input_images)
        image9 = self.warp_layer(t0, ddf9)
        ddf8 = self.u08(torch.cat([t0, image9], 1))  # + ddf9
        image8 = self.warp_layer(t0, ddf8)
        ddf7 = self.u07(torch.cat([t0, image9, image8], 1))  # + ddf8
        image7 = self.warp_layer(t0, ddf7)
        ddf6 = self.u06(torch.cat([t0, image9, image8, image7], 1))  # + ddf7
        image6 = self.warp_layer(t0, ddf6)

        ddf5_1 = self.u05(torch.cat([t0, image1, image2, image3, image4], 1))  # + ddf4
        ddf5_2 = self.u05(torch.cat([t0, image9, image8, image7, image6], 1))  # + ddf6
        ddf5 = (ddf5_1 + ddf5_2) / 2
        image5 = self.warp_layer(t0, ddf5)

        return (
            ddf1,
            ddf2,
            ddf3,
            ddf4,
            ddf5,
            ddf6,
            ddf7,
            ddf8,
            ddf9,
            image1,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
        )


if __name__ == "__main__":
    cbamunet = FNet()
    x = torch.randn(2, 1, 128, 128, 64)
    y = cbamunet(x)
    print(y)
    print(y[0].size())
