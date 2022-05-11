import numpy as np
import ants
import os
from time import time
import shutil
def get_transform(input_file, target_file,type_of_transform,save_path):
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "80"
        os.environ["ANTS_RANDOM_SEED"] = "3"
        begin_time = time()
        os.makedirs(save_path,exist_ok=True)
        moving = ants.image_read(target_file)
        fixed = ants.image_read(input_file)
        
        filename = input_file.split("/")[-1].split(".")[-2]
        print(f"\t-> Registration.")
        reg = ants.registration(
            fixed=fixed, moving=moving, type_of_transform=type_of_transform)

        moved = reg["warpedmovout"]
        end_time = time()
        run_time = end_time-begin_time
        print(f"\t-> Used:{run_time}s")
        
        print(reg)
        for output_file in reg["fwdtransforms"]:
            if "nii" in output_file:
                shutil.move(output_file, f"{save_path}/{filename}_Warp.nii.gz")
            if "mat" in output_file:
                shutil.move(output_file, f"{save_path}/{filename}_GenericAffine.mat")
        return output_file


files = os.listdir("/home/zhaosheng/4dct_test_nii/")
patients_dict = {}

for patient in files:
    sub_path = os.path.join("/home/zhaosheng/4dct_test_nii/",patient)
    patients_dict[patient]=[_file.split("_")[-1].split(".")[0] for _file in os.listdir(sub_path)]

p_num = 0
print(patients_dict)
for patient in patients_dict:
    try:
        if len(patients_dict[patient]) == 10:
            p_num+=1
            for i in range(1,10):
                transform_path = get_transform(f"/home/zhaosheng/4dct_test_nii/{patient}/{patient}_t{str(i)}.nii",
                                                 f"/home/zhaosheng/4dct_test_nii/{patient}/{patient}_t0.nii","SyNAggro",
                                                 "/home/zhaosheng/4dct_test_nii_transform_SyNAggro/",
                                                 )
    except Exception as e:
        print(e)