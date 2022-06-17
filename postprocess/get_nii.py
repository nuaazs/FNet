import ants
import numpy as np
import os
from IPython import embed
npy_path = "/mnt/zhaosheng/FNet/results/cbamunet_resunits2_003/349/images"
epoch="349"
nii_save_path = "/mnt/zhaosheng/FNet/postprocess/output_niis"
real_nii_path = "/mnt/zhaosheng/4dct/resampled"
npys = [os.path.join(npy_path,_file) for _file in  os.listdir(npy_path) if "real" not in _file and "_t5" in _file]
print(npys)
print(len(npys))
for npy_file in npys:
    
    p_index = npy_file.split("/")[-1].split(".")[0].split("_")[1]
    
   
    input_nii = os.path.join(real_nii_path,f"{p_index}_t0_resampled.nii")
    input_nii_image = ants.image_read(input_nii)
    #input_nii_image.to_file(os.path.join(nii_save_path,f"{epoch}_{p_index}_input.nii.gz"))
    

    t0_npy = os.path.join(npy_path,f"{epoch}_{p_index}_t0_real.npy")
    t0_array = np.load(t0_npy,allow_pickle=True)[0,0]
    t0_nii_image = input_nii_image.new_image_like(t0_array)
    t0_nii_image.to_file(os.path.join(nii_save_path,f"{epoch}_{p_index}_t0_real.nii.gz"))


    for t in range(1,10):
        t_index = f"t{t}"
        # t_index = npy_file.split("/")[-1].split(".")[0].split("_")[2]

        real_npy = os.path.join(npy_path,f"{epoch}_{p_index}_{t_index}_real.npy")
        real_npy_array = np.load(real_npy,allow_pickle=True)[0,0]
        real_nii_image = input_nii_image.new_image_like(real_npy_array)
        real_nii_image.to_file(os.path.join(nii_save_path,f"{epoch}_{p_index}_{t_index}_real.nii.gz"))
        

        fake_npy = os.path.join(npy_path,f"{epoch}_{p_index}_{t_index}.npy")
        npy_array = np.load(fake_npy,allow_pickle=True)[0,0]#*(285+3770)-285 #.transpose(1,2,0)
        fake_nii_image = input_nii_image.new_image_like(npy_array)
        fake_nii_image.to_file(os.path.join(nii_save_path,f"{epoch}_{p_index}_{t_index}_fake.nii.gz"))

    break
