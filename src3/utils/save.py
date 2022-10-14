import numpy as np
import os
import monai
import ants


def save_nii(tensor, filename, save_npy_path):
    npy = tensor.cpu().detach().numpy()
    np.save(os.path.join(save_npy_path, filename), npy)
