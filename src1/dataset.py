import os
import numpy as np
import torch
from monai.data import DataLoader, Dataset, CacheDataset
import random
import monai

seed = 123456
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
monai.utils.set_determinism(seed=seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandAffined,
    Resized,
    ScaleIntensityRanged,
    EnsureTyped,
    RandZoomd,
    RandRotate90d,
    RandFlipd,
    SaveImaged,
    AsChannelFirstd,
    SqueezeDimd,
)


def getDataLoader(
    batch_size=1,
    num_workers=5,
    istry=False,
    mode="train",
    ddf_prefix="49",
    ddf_dir="/mnt/zhaosheng/FNet/data/ddfs",
    img_dir="/mnt/zhaosheng/4dct/resampled",
):

    data_inputs_reample = []
    for item in [_file.split("_")[0] for _file in os.listdir(img_dir) if "t9" in _file]:
        if item not in data_inputs_reample:
            data_inputs_reample.append(item)

    data_inputs = []

    for item in [_file.split("_")[0] for _file in os.listdir(img_dir) if "t9" in _file]:
        if (item not in data_inputs) and (item in data_inputs_reample):
            add = True
            for tt in range(10):
                if not os.path.exists(
                    os.path.join(img_dir, f"{item}_t{tt}_resampled.nii")
                ):
                    add = False
            if add:
                data_inputs.append(item)

    data_dicts = [
        {
            "t0": os.path.join(img_dir, f"{idx}_t0_resampled.nii"),
            "t5": os.path.join(img_dir, f"{idx}_t5_resampled.nii"),
            "ddf": os.path.join(ddf_dir, f"{ddf_prefix}_{idx}_5_0.nii"),
            "pid": f"{idx}",
        }
        for idx in sorted(data_inputs)
    ]

    if istry:
        train_files, val_files = data_dicts[:10], data_dicts[-10:]
    else:
        total_length = len(data_inputs)
        # train_files, val_files = data_dicts[:-1*int(total_length/5)], data_dicts[-1*int(total_length/5):]
        train_files, val_files = data_dicts, data_dicts[-1 * int(total_length / 5) :]
        print(
            f"Total data: {total_length} patients. Used {total_length-int(total_length/5)} for train and {int(total_length/5)} for test."
        )
    train_transforms = Compose(
        [
            LoadImaged(keys=["t0", "t5", "ddf"],),
            AddChanneld(keys=["t0", "t5"],),
            ScaleIntensityRanged(
                keys=["t0", "t5",],
                a_min=-285,
                a_max=3770,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["t0", "t5", "ddf"],),
        ]
    )
    if mode == "train":
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=0.5,
            num_workers=num_workers,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=train_transforms,
            cache_rate=0.5,
            num_workers=num_workers,
        )
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
        return train_loader, val_loader
    else:
        val_ds = CacheDataset(
            data=val_files,
            transform=train_transforms,
            cache_rate=0.5,
            num_workers=num_workers,
        )
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
        return 0, val_loader


if __name__ == "__main__":
    train_loader, val_loader = getDataLoader(batch_size=1, num_workers=0, istry=True)
    for batch_data in train_loader:
        t0_image = batch_data["t0"].cuda()
        ddf1 = batch_data["ddf"].cuda()
        print(t0_image.shape)
        print(ddf1.shape)
