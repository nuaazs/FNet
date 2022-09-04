from antspynet.utilities import lung_extraction
import numpy as np
import torch
import ants
from utils.log import Logger
from monai.transforms import KeepLargestConnectedComponent
check_log = Logger('check.log')

# !TODO :添加保留最大体积 MONAI
#klcc = KeepLargestConnectedComponent(is_onehot=None, independent=True, connectivity=None)
#  The input is assumed to be a channel-first PyTorch Tensor:
# 1) For not OneHot format data, the values correspond to expected labels, 0 will be treated as background and the over-segment pixels will be set to 0. 2) For OneHot format data, the values should be 0, 1 on each labels, the over-segment pixels will be set to 0 in its channel.
# For example: Use with applied_labels=[1], is_onehot=False, connectivity=1:


def compute_volume(img):
    """
    Computes the area (number of pixels) of a binary mask and multiplies the pixels
    with the pixel dimension of the acquired CT image
    Args:
        lung_mask: binary lung mask
        pixdim: list or tuple with two values
    Returns: the lung area in mm^2
    """
    pixdim = img.spacing
    img = img.numpy()
    # img[img >= 0.5] = 1
    # img[img < 0.5] = 0
    lung_pixels = np.sum(img)
    return lung_pixels * pixdim[0] * pixdim[1]* pixdim[2]

def do_klcc(img):
    # img size -> (512,512,121)
    tensor_ = torch.tensor(img.numpy())
    tensor_ = torch.unsqueeze(tensor_, dim=0)
    print(tensor_.shape)
    output = klcc(tensor_)
    npy = output.numpy()[0]
    print(npy.shape)
    img_after = img.new_image_like(npy)
    return img_after

def get_lung_mask(img):
    lung_result = lung_extraction(img, modality='ct', verbose=False)
    left_lung = lung_result['probability_images'][1]
    right_lung = lung_result['probability_images'][2]
    check_log.info(f"\t\t-> start klcc.")

    # left_lung= do_klcc(left_lung)
    # right_lung= do_klcc(right_lung)

    check_log.info(f"\t\t-> end klcc.")
    left_lung_volume = compute_volume(left_lung)
    right_lung_volume = compute_volume(right_lung)
    return {"left_lung_volume":left_lung_volume,
            "right_lung_volume":right_lung_volume,
            "left_lung":left_lung,
            "right_lung":right_lung}

def get_loc(_npy):
    _npy = np.array(_npy)
    loc = np.mean(np.argwhere(_npy == 1),axis=0)
    return loc

if __name__ == "__main__":
    img = ants.image_read("/media/wurenyao/TOSHIBA EXT/4dct_512_resampled_fake/356857_t8_fake_fake.nii.gz")
    img2 = img.new_image_like(np.ones((512,512,121)))
    print(img2)
    img_after = do_klcc(img2)
    print(img_after)
    (img+1000).plot()
    (img_after+1000).plot()