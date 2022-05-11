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
    Invertd,
)


def getDataLoader_pre(batch_size=1,num_workers=5,istry=False):

    test_data_dir = "/zhaosheng_data/4dct_4_test"
    test_mask_dir = "/dataset1/4dct_4_lungs_test"
    test_data_inputs = []
    for pname in os.listdir(test_data_dir):
        for item in [_file.split("_")[0] for _file in os.listdir(os.path.join(test_data_dir,pname)) if "t9" in _file]:
            if item not in test_data_inputs:
                test_data_inputs.append(item)

    
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
    
    
    
    val_files = test_dicts


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

    val_ds = CacheDataset(data=val_files, transform=val_transforms,cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
    return val_loader,val_transforms