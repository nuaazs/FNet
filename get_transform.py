from time import *
import ants
import shutil
import os
from tqdm import tqdm
def get_transform(moving_file,fixed_file,type_of_transform="SyNAggro",plots_path="/zhaosheng_data/4dct_2_transform_png",transform_path="/zhaosheng_data/4dct_4_transform"):
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "80"
    os.environ["ANTS_RANDOM_SEED"] = "3"
    os.makedirs(plots_path,exist_ok=True)
    os.makedirs(transform_path,exist_ok=True)
    moving = ants.image_read(moving_file)
    fixed = ants.image_read(fixed_file)
    begin_time = time()
    filename = fixed_file.split("/")[-1].split(".")[-2]
    reg = ants.registration(
        fixed=fixed, moving=moving, type_of_transform=type_of_transform)

    moved = reg["warpedmovout"]
    moved2 = moved +1000
    moved2.plot(title='moved', axis=1, cbar=True,
            filename=os.path.join(plots_path, filename+"_moved.png"))

    ants.plot(moved2, overlay=fixed, overlay_cmap='hot', overlay_alpha=0.5,
            axis=1, cbar=True, filename=os.path.join(plots_path, filename+"_overlay_1.png"))
    ants.plot(moved2, overlay=fixed, overlay_cmap='hot', overlay_alpha=0.5,
            axis=0, cbar=True, filename=os.path.join(plots_path, filename+"_overlay_0.png"))
    # SAVE transform

    end_time = time()
    run_time = end_time-begin_time



    for output_file in reg["fwdtransforms"]:
        if "nii" in output_file:
            shutil.move(output_file, f"{transform_path}/{filename}_Warp.nii.gz")
        if "mat" in output_file:
            shutil.move(output_file, f"{transform_path}/{filename}_GenericAffine.mat")
        return output_file

if __name__ == "__main__":

    files = sorted([os.path.join("/dataset1/4dct_4",_file) for _file in os.listdir("/dataset1/4dct_4") if "_t9" in _file])
    for file in tqdm(files):
        filename = file.split("/")[-1].split("_")[0]
        for i in range(1,10):
            try:
                get_transform(f"/dataset1/4dct_4/{filename}_t0.nii",f"/dataset1/4dct_4/{filename}_t{i}.nii",transform_path="/home/zhaosheng/4dct_4_transform_SyNAggro",plots_path="/home/zhaosheng/4dct_4_transform_SyNAggro_pngs")
            except:
                print(f"Pass {filename} {i}")
