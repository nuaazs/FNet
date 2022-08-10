import ants
import numpy as np
import os
from IPython import embed
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str, default='/mnt/zhaosheng/FNet/ddfs',help='')
args = parser.parse_args()

files = sorted([os.path.join(args.path,_file) for _file in os.listdir(args.path)])
for file in tqdm(files):
    _array = np.load(file)[0]
    print(_array.shape)
    image = ants.from_numpy(_array)
    image.to_file(file.replace(".npy",".nii"))
