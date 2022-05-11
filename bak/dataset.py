import os
import numpy as np
import torch
from monai.data import DataLoader, Dataset, CacheDataset
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


def getDataLoader(batch_size=1,num_workers=5,istry=False):
    data_dir = "/dataset1/4dct_0510/resampled"
    # test_data_dir = "/zhaosheng_data/4dct_4_test"
    transform_dir = "/dataset1/4dct_0510/transform"
    # test_trans_dir = "/dataset1/4dct_4_lungs_test"
    
    data_inputs = []
    for item in [_file.split("_")[0] for _file in os.listdir(data_dir) if "t9" in _file]:
        if item not in data_inputs:
            data_inputs.append(item)

    data_dicts = [
        {
            "t0_image": os.path.join(data_dir,f"{idx}_t0_resampled.nii"),
            "t1_image": os.path.join(data_dir,f"{idx}_t1_resampled.nii"),
            "t2_image": os.path.join(data_dir,f"{idx}_t2_resampled.nii"),
            "t3_image": os.path.join(data_dir,f"{idx}_t3_resampled.nii"),
            "t4_image": os.path.join(data_dir,f"{idx}_t4_resampled.nii"),
            "t5_image": os.path.join(data_dir,f"{idx}_t5_resampled.nii"),
            "t6_image": os.path.join(data_dir,f"{idx}_t6_resampled.nii"),
            "t7_image": os.path.join(data_dir,f"{idx}_t7_resampled.nii"),
            "t8_image": os.path.join(data_dir,f"{idx}_t8_resampled.nii"),
            "t9_image": os.path.join(data_dir,f"{idx}_t9_resampled.nii"),
            "t1_trans": os.path.join(transform_dir,f"{idx}_t1_Warp.nii.gz"),
            "t2_trans": os.path.join(transform_dir,f"{idx}_t2_Warp.nii.gz"),
            "t3_trans": os.path.join(transform_dir,f"{idx}_t3_Warp.nii.gz"),
            "t4_trans": os.path.join(transform_dir,f"{idx}_t4_Warp.nii.gz"),
            "t5_trans": os.path.join(transform_dir,f"{idx}_t5_Warp.nii.gz"),
            "t6_trans": os.path.join(transform_dir,f"{idx}_t6_Warp.nii.gz"),
            "t7_trans": os.path.join(transform_dir,f"{idx}_t7_Warp.nii.gz"),
            "t8_trans": os.path.join(transform_dir,f"{idx}_t8_Warp.nii.gz"),
            "t9_trans": os.path.join(transform_dir,f"{idx}_t9_Warp.nii.gz"),
            "pid": f"{idx}",
        }
        for idx in sorted(data_inputs)
    ]

    if istry:
        train_files, val_files = data_dicts[:10], data_dicts[-10:]
    else:
        train_files, val_files = data_dicts[:-20], data_dicts[-20:]

    train_transforms = Compose(
        [
            LoadImaged(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t1_trans","t2_trans","t3_trans","t4_trans","t5_trans","t6_trans","t7_trans","t8_trans","t9_trans"],
            ),
            AddChanneld(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image"],
            ),
            AsChannelFirstd(
                keys=["t1_trans","t2_trans","t3_trans","t4_trans","t5_trans","t6_trans","t7_trans","t8_trans","t9_trans"],channel_dim=-1, allow_missing_keys=False
            ),
            SqueezeDimd(
                keys=["t1_trans","t2_trans","t3_trans","t4_trans","t5_trans","t6_trans","t7_trans","t8_trans","t9_trans"],dim=-1, allow_missing_keys=False
            ),
            ScaleIntensityRanged(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image"],
                a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True,
            ),
            EnsureTyped(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t1_trans","t2_trans","t3_trans","t4_trans","t5_trans","t6_trans","t7_trans","t8_trans","t9_trans"],
            ),
        ]
    )
    
    train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_ds = CacheDataset(data=val_files, transform=train_transforms,cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
    return train_loader,val_loader

if __name__ == "__main__":
    train_loader,val_loader = getDataLoader(batch_size=1,num_workers=5,istry=True)

    for batch_data in train_loader:
        t0_image = batch_data["t0_image"].cuda()
        t1_trans = batch_data["t1_trans"].cuda()
        print(t0_image.shape)
        print(t1_trans.shape)