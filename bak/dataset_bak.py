import os
import numpy as np
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
)


def getDataLoader(batch_size=1,num_workers=5,istry=False):
    data_dir = "/dataset1/4dct_4"
    test_data_dir = "/zhaosheng_data/4dct_4_test"
    mask_dir = "/dataset1/4dct_4_lungs"
    test_mask_dir = "/dataset1/4dct_4_lungs_test"
    data_inputs = []
    for item in [_file.split("_")[0] for _file in os.listdir(data_dir) if "t9" in _file]:
        if item not in data_inputs:
            data_inputs.append(item)


    test_data_inputs = []
    for pname in os.listdir(test_data_dir):
        for item in [_file.split("_")[0] for _file in os.listdir(os.path.join(test_data_dir,pname)) if "t9" in _file]:
            if item not in test_data_inputs:
                test_data_inputs.append(item)
    data_dicts = [
        {
            "t0_image": os.path.join(data_dir,f"{idx}_t0.nii"),
            "t1_image": os.path.join(data_dir,f"{idx}_t1.nii"),
            "t2_image": os.path.join(data_dir,f"{idx}_t2.nii"),
            "t3_image": os.path.join(data_dir,f"{idx}_t3.nii"),
            "t4_image": os.path.join(data_dir,f"{idx}_t4.nii"),
            "t5_image": os.path.join(data_dir,f"{idx}_t5.nii"),
            "t6_image": os.path.join(data_dir,f"{idx}_t6.nii"),
            "t7_image": os.path.join(data_dir,f"{idx}_t7.nii"),
            "t8_image": os.path.join(data_dir,f"{idx}_t8.nii"),
            "t9_image": os.path.join(data_dir,f"{idx}_t9.nii"),
            "t0_mask": os.path.join(mask_dir,f"{idx}_t0_lung.nii"),
            "t1_mask": os.path.join(mask_dir,f"{idx}_t1_lung.nii"),
            "t2_mask": os.path.join(mask_dir,f"{idx}_t2_lung.nii"),
            "t3_mask": os.path.join(mask_dir,f"{idx}_t3_lung.nii"),
            "t4_mask": os.path.join(mask_dir,f"{idx}_t4_lung.nii"),
            "t5_mask": os.path.join(mask_dir,f"{idx}_t5_lung.nii"),
            "t6_mask": os.path.join(mask_dir,f"{idx}_t6_lung.nii"),
            "t7_mask": os.path.join(mask_dir,f"{idx}_t7_lung.nii"),
            "t8_mask": os.path.join(mask_dir,f"{idx}_t8_lung.nii"),
            "t9_mask": os.path.join(mask_dir,f"{idx}_t9_lung.nii"),
            "pid": f"{idx}",
        }
        for idx in sorted(data_inputs)
    ]
    
    test_dicts = [
        {
            "t0_image": os.path.join(test_data_dir,f"{idx}/{idx}_t0.nii"),
            "t1_image": os.path.join(test_data_dir,f"{idx}/{idx}_t1.nii"),
            "t2_image": os.path.join(test_data_dir,f"{idx}/{idx}_t2.nii"),
            "t3_image": os.path.join(test_data_dir,f"{idx}/{idx}_t3.nii"),
            "t4_image": os.path.join(test_data_dir,f"{idx}/{idx}_t4.nii"),
            "t5_image": os.path.join(test_data_dir,f"{idx}/{idx}_t5.nii"),
            "t6_image": os.path.join(test_data_dir,f"{idx}/{idx}_t6.nii"),
            "t7_image": os.path.join(test_data_dir,f"{idx}/{idx}_t7.nii"),
            "t8_image": os.path.join(test_data_dir,f"{idx}/{idx}_t8.nii"),
            "t9_image": os.path.join(test_data_dir,f"{idx}/{idx}_t9.nii"),
            "t0_mask": os.path.join(test_mask_dir,f"{idx}_t0_lung.nii"),
            "t1_mask": os.path.join(test_mask_dir,f"{idx}_t1_lung.nii"),
            "t2_mask": os.path.join(test_mask_dir,f"{idx}_t2_lung.nii"),
            "t3_mask": os.path.join(test_mask_dir,f"{idx}_t3_lung.nii"),
            "t4_mask": os.path.join(test_mask_dir,f"{idx}_t4_lung.nii"),
            "t5_mask": os.path.join(test_mask_dir,f"{idx}_t5_lung.nii"),
            "t6_mask": os.path.join(test_mask_dir,f"{idx}_t6_lung.nii"),
            "t7_mask": os.path.join(test_mask_dir,f"{idx}_t7_lung.nii"),
            "t8_mask": os.path.join(test_mask_dir,f"{idx}_t8_lung.nii"),
            "t9_mask": os.path.join(test_mask_dir,f"{idx}_t9_lung.nii"),
            "pid": f"{idx}",
        }
        for idx in sorted(test_data_inputs)
    ]
    
    
    if istry:
        train_files, val_files = data_dicts[:10], test_dicts
    else:
        train_files, val_files = data_dicts+test_dicts, test_dicts

    train_transforms = Compose(
        [
            LoadImaged(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
            ),
            AddChanneld(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
            ),
            ScaleIntensityRanged(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
                a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True,
            ),
            #RandAffined(
            #    keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
            #            "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
            #    mode=('bilinear', 'bilinear','bilinear', 'bilinear','bilinear', 'bilinear','bilinear', 'bilinear','bilinear', 'bilinear',
            #            'bilinear', 'bilinear','bilinear', 'bilinear','bilinear', 'bilinear','bilinear', 'bilinear','bilinear', 'bilinear'),
            #    prob=1.0, spatial_size=(128, 128, 64),
            #    rotate_range=(0, 0, np.pi / 15), scale_range=(0.1, 0.1, 0.1),
            #    padding_mode='zeros',
            #    cache_grid=True
            #),
            Resized(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
                spatial_size=(128,128,64)
            ),

            
            # RandZoomd(
            #    keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
            #            "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
            #    prob=0.5, min_zoom=0.9, max_zoom=1.1
            # ),

            # RandRotate90d(
            #    keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
            #            "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
            #    prob=0.1
            # ),
            #RandFlipd(
            #   keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
            #           "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
            #   prob=0.1
            #),
            EnsureTyped(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
            ),
        ]
    )
    val_transforms = Compose([
            LoadImaged(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
            ),
            AddChanneld(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
            ),
            ScaleIntensityRanged(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
                a_min=-1000, a_max=2000,b_min=0.0, b_max=1.0,
                clip=True,
            ),
            Resized(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
                spatial_size=(128,128,64)
            ),
            EnsureTyped(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t0_mask","t1_mask","t2_mask","t3_mask","t4_mask","t5_mask","t6_mask","t7_mask","t8_mask","t9_mask"],
            ),
            # SaveImaged(
            #     keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image"],
            #     output_dir='./preprocessed_val_data',
            #     output_postfix='trans',
            #     output_ext='.nii.gz',
            #     resample=False,
            #     scale=None, 
            #     squeeze_end_dims=True,
            #     separate_folder=True,
            #     print_log=True
            # )
        ])
    train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms,cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
    return train_loader,val_loader



def getDataLoader_ddf(batch_size=1,num_workers=5,istry=False):
    data_dir = "/dataset1/4dct_4"
    test_data_dir = "/zhaosheng_data/4dct_4_test"


    ddf_dir = "/home/zhaosheng/4dct_4_transform_SyNAggro"
    test_ddf_dir = "/dataset1/4dct_4_ddfs_test"


    data_inputs = []
    for item in [_file.split("_")[0] for _file in os.listdir(data_dir) if "t9" in _file]:
        if item not in data_inputs:
            data_inputs.append(item)


    test_data_inputs = []
    for pname in os.listdir(test_data_dir):
        for item in [_file.split("_")[0] for _file in os.listdir(os.path.join(test_data_dir,pname)) if "t9" in _file]:
            if item not in test_data_inputs:
                test_data_inputs.append(item)
    data_dicts = [
        {
            "t0_image": os.path.join(data_dir,f"{idx}_t0.nii"),
            "t1_image": os.path.join(data_dir,f"{idx}_t1.nii"),
            "t2_image": os.path.join(data_dir,f"{idx}_t2.nii"),
            "t3_image": os.path.join(data_dir,f"{idx}_t3.nii"),
            "t4_image": os.path.join(data_dir,f"{idx}_t4.nii"),
            "t5_image": os.path.join(data_dir,f"{idx}_t5.nii"),
            "t6_image": os.path.join(data_dir,f"{idx}_t6.nii"),
            "t7_image": os.path.join(data_dir,f"{idx}_t7.nii"),
            "t8_image": os.path.join(data_dir,f"{idx}_t8.nii"),
            "t9_image": os.path.join(data_dir,f"{idx}_t9.nii"),
            "t1_ddf": os.path.join(ddf_dir,f"{idx}_t1_Warp.nii.gz"),
            "t2_ddf": os.path.join(ddf_dir,f"{idx}_t2_Warp.nii.gz"),
            "t3_ddf": os.path.join(ddf_dir,f"{idx}_t3_Warp.nii.gz"),
            "t4_ddf": os.path.join(ddf_dir,f"{idx}_t4_Warp.nii.gz"),
            "t5_ddf": os.path.join(ddf_dir,f"{idx}_t5_Warp.nii.gz"),
            "t6_ddf": os.path.join(ddf_dir,f"{idx}_t6_Warp.nii.gz"),
            "t7_ddf": os.path.join(ddf_dir,f"{idx}_t7_Warp.nii.gz"),
            "t8_ddf": os.path.join(ddf_dir,f"{idx}_t8_Warp.nii.gz"),
            "t9_ddf": os.path.join(ddf_dir,f"{idx}_t9_Warp.nii.gz"),
            "pid": f"{idx}",
        }
        for idx in sorted(data_inputs)
    ]

    if istry:
        train_files, val_files = data_dicts[:10], data_dicts[-10:]
    else:
        train_files, val_files = data_dicts,data_dicts

    train_transforms = Compose(
        [
            LoadImaged(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t1_ddf","t2_ddf","t3_ddf","t4_ddf","t5_ddf","t6_ddf","t7_ddf","t8_ddf","t9_ddf"],
            ),
            AddChanneld(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image"],
            ),
            ScaleIntensityRanged(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image"],
                a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True,
            ),

            Resized(
                keys=["t1_ddf","t2_ddf","t3_ddf","t4_ddf","t5_ddf","t6_ddf","t7_ddf","t8_ddf","t9_ddf"],
                spatial_size=(3,128,128,64)
            ),
            Resized(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image"],
                spatial_size=(128,128,64)
            ),
            EnsureTyped(
                keys=["t0_image","t1_image","t2_image","t3_image","t4_image","t5_image","t6_image","t7_image","t8_image","t9_image",
                        "t1_ddf","t2_ddf","t3_ddf","t4_ddf","t5_ddf","t6_ddf","t7_ddf","t8_ddf","t9_ddf"],
            ),
        ]
    )
    
    train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_ds = CacheDataset(data=val_files, transform=train_transforms,cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
    return train_loader,val_loader
