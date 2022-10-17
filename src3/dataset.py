import os
import numpy as np
import torch
from monai.data import DataLoader, Dataset, CacheDataset
import random
import monai

from preprocess.load_data import get_patients_withtumor

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
    img_dir="/mnt/zhaosheng/4dct/output",
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
                if not os.path.exists(os.path.join(img_dir, f"{item}_t{tt}.nii")):
                    add = False
            if add:
                data_inputs.append(item)
    with_tumors = get_patients_withtumor()
    data_dicts = [
        {
            "t0_image": os.path.join(img_dir, f"{idx}_t0.nii"),
            "t1_image": os.path.join(img_dir, f"{idx}_t1.nii"),
            "t2_image": os.path.join(img_dir, f"{idx}_t2.nii"),
            "t3_image": os.path.join(img_dir, f"{idx}_t3.nii"),
            "t4_image": os.path.join(img_dir, f"{idx}_t4.nii"),
            "t5_image": os.path.join(img_dir, f"{idx}_t5.nii"),
            "t6_image": os.path.join(img_dir, f"{idx}_t6.nii"),
            "t7_image": os.path.join(img_dir, f"{idx}_t7.nii"),
            "t8_image": os.path.join(img_dir, f"{idx}_t8.nii"),
            "t9_image": os.path.join(img_dir, f"{idx}_t9.nii"),
            "ddf1_image": os.path.join(ddf_dir, f"{ddf_prefix}_{idx}_1_0.nii"),
            "ddf2_image": os.path.join(ddf_dir, f"{ddf_prefix}_{idx}_2_0.nii"),
            "ddf3_image": os.path.join(ddf_dir, f"{ddf_prefix}_{idx}_3_0.nii"),
            "ddf4_image": os.path.join(ddf_dir, f"{ddf_prefix}_{idx}_4_0.nii"),
            "ddf5_image": os.path.join(ddf_dir, f"{ddf_prefix}_{idx}_5_0.nii"),
            "ddf6_image": os.path.join(ddf_dir, f"{ddf_prefix}_{idx}_6_0.nii"),
            "ddf7_image": os.path.join(ddf_dir, f"{ddf_prefix}_{idx}_7_0.nii"),
            "ddf8_image": os.path.join(ddf_dir, f"{ddf_prefix}_{idx}_8_0.nii"),
            "ddf9_image": os.path.join(ddf_dir, f"{ddf_prefix}_{idx}_9_0.nii"),
            "pid": f"{idx}",
        }
        for idx in sorted(data_inputs) if idx in with_tumors
    ]

    if istry:
        train_files, val_files = data_dicts[:10], data_dicts[-10:]
    else:
        total_length = len(data_dicts)
        # train_files, val_files = data_dicts[:-1*int(total_length/5)], data_dicts[-1*int(total_length/5):]
        train_files, val_files = data_dicts, data_dicts[-1 * int(total_length / 5) :]
        print(
            f"Total data: {total_length} patients. Used {total_length-int(total_length/5)} for train and {int(total_length/5)} for test."
        )
    train_transforms = Compose(
        [
            LoadImaged(
                keys=[
                    "t0_image",
                    "t1_image",
                    "t2_image",
                    "t3_image",
                    "t4_image",
                    "t5_image",
                    "t6_image",
                    "t7_image",
                    "t8_image",
                    "t9_image",
                    "ddf1_image",
                    "ddf2_image",
                    "ddf3_image",
                    "ddf4_image",
                    "ddf5_image",
                    "ddf6_image",
                    "ddf7_image",
                    "ddf8_image",
                    "ddf9_image",
                ],
            ),
            AddChanneld(
                keys=[
                    "t0_image",
                    "t1_image",
                    "t2_image",
                    "t3_image",
                    "t4_image",
                    "t5_image",
                    "t6_image",
                    "t7_image",
                    "t8_image",
                    "t9_image",
                ],
            ),
            ScaleIntensityRanged(
                keys=[
                    "t0_image",
                    "t1_image",
                    "t2_image",
                    "t3_image",
                    "t4_image",
                    "t5_image",
                    "t6_image",
                    "t7_image",
                    "t8_image",
                    "t9_image",
                ],
                a_min=-285,
                a_max=3770,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(
                keys=[
                    "t0_image",
                    "t1_image",
                    "t2_image",
                    "t3_image",
                    "t4_image",
                    "t5_image",
                    "t6_image",
                    "t7_image",
                    "t8_image",
                    "t9_image",
                    "ddf1_image",
                    "ddf2_image",
                    "ddf3_image",
                    "ddf4_image",
                    "ddf5_image",
                    "ddf6_image",
                    "ddf7_image",
                    "ddf8_image",
                    "ddf9_image",
                ],
            ),
        ]
    )
    if mode == "train":
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=1,
            num_workers=num_workers,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=train_transforms,
            cache_rate=1,
            num_workers=num_workers,
        )
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
        return train_loader, val_loader
    else:
        val_ds = CacheDataset(
            data=val_files,
            transform=train_transforms,
            cache_rate=1,
            num_workers=num_workers,
        )
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
        return 0, val_loader


if __name__ == "__main__":
    train_loader, val_loader = getDataLoader(batch_size=1, num_workers=0, istry=True)
    for batch_data in train_loader:
        t0_image = batch_data["t0_image"].cuda()
        ddf1 = batch_data["ddf1_image"].cuda()
        print(t0_image.shape)
        print(ddf1.shape)
