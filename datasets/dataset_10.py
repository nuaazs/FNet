import os
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


def getDataLoader(batch_size=1, num_workers=0, istry=False, mode="train", t_index=1):
    data_dir = "/mnt/zhaosheng/4dct/resampled"
    ddf_dir = "/mnt/zhaosheng/FNet/ddfs"
    data_inputs_reample = []
    for item in [
        _file.split("_")[0] for _file in os.listdir(data_dir) if "t9" in _file
    ]:
        if item not in data_inputs_reample:
            data_inputs_reample.append(item)

    data_inputs = []
    for item in [
        _file.split("_")[0] for _file in os.listdir(data_dir) if "t9" in _file
    ]:
        if (item not in data_inputs) and (item in data_inputs_reample):
            add = True
            for tt in range(10):
                if not os.path.exists(
                    os.path.join(data_dir, f"{item}_t{tt}_resampled.nii")
                ):
                    add = False
                ddf_file = os.path.join(ddf_dir, f"49_{item}_{tt}_0.nii")
                if not (tt == 0 or os.path.exists(ddf_file)):
                    add = False
            if add:
                data_inputs.append(item)
    data_dicts = []

    for idx in sorted(data_inputs):
        image_keys = []
        ddf_keys = []
        tiny = {}
        for tt in range(t_index + 1):
            tiny[f"t{tt}_image"] = os.path.join(data_dir, f"{idx}_t{tt}_resampled.nii")
            image_keys.append(f"t{tt}_image")
            if tt > 0:
                tiny[f"t{tt}_ddf"] = os.path.join(ddf_dir, f"49_{idx}_{tt}_0.nii")
                ddf_keys.append(f"t{tt}_ddf")
        tiny["pid"] = f"{idx}"
        data_dicts.append(tiny)

    if istry:
        train_files, val_files = data_dicts[:10], data_dicts[-10:]
    else:
        total_length = len(data_inputs)
        train_files, val_files = (
            data_dicts[: -1 * int(total_length / 5)],
            data_dicts[-1 * int(total_length / 5) :],
        )
        # train_files, val_files = data_dicts,data_dicts[-1*int(total_length/5):]
        print(
            f"Total data: {total_length} patients. Used {total_length-int(total_length/5)} for train and {int(total_length/5)} for test."
        )
    image_keys = sorted([f"t{tt}_image" for tt in range(t_index + 1)])
    ddf_keys = sorted([f"t{tt}_ddf" for tt in range(1, t_index + 1)])
    train_transforms = Compose(
        [
            LoadImaged(keys=image_keys + ddf_keys,),
            AddChanneld(keys=image_keys,),
            ScaleIntensityRanged(
                keys=image_keys,
                a_min=-285,
                a_max=3770,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=image_keys + ddf_keys,),
        ]
    )
    if mode == "train":
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=num_workers,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=num_workers,
        )
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
        return train_loader, val_loader
    else:
        val_ds = CacheDataset(
            data=val_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=num_workers,
        )
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
        return 0, val_loader


if __name__ == "__main__":
    t_index = 3
    train_loader, val_loader = getDataLoader(
        batch_size=1, num_workers=0, istry=True, t_index=t_index
    )
    for batch_data in train_loader:
        t0_image = batch_data["t0_image"]
        print(f"t0 shape: {t0_image.shape}")
        for tt in range(1, t_index + 1):
            ddf = batch_data[f"t{tt}_ddf"]
            img = batch_data[f"t{tt}_image"]
            print(f"ddf{tt} shape: {ddf.shape}")
            print(f"img t{tt} shape: {img.shape}")
        break
