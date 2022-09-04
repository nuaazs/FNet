import ants
import numpy as np
import os
from IPython import embed

def resample_3d_npy(npy_array,shape):
    ddf_1 = []
    for index in range(3):
        ddf_npy_1 = npy_array[0,index]
        ddf_raw_1 = ants.from_numpy(ddf_npy_1)
        ddf_raw_1 = ants.resample_image(ddf_raw_1,shape,True,4)
        ddf_1.append(ddf_raw_1.numpy())
    ddf_1 = np.array(ddf_1)
    return ddf_1


net1_epoch = "14"
net1_name = "cbamunet_resunits2_A2B_ft"
net1_npy_path = f"/mnt/zhaosheng/FNet/results/{net1_name}/{net1_epoch}/images"

net2_epoch = "14"
net2_name = "cbamunet_resunits2_B2A_ft"
net2_npy_path = f"/mnt/zhaosheng/FNet/results/{net2_name}/{net2_epoch}/images"

nii_save_path = "/mnt/zhaosheng/FNet/postprocess/output_niis"
nii_512_save_path = "/mnt/zhaosheng/FNet/postprocess/output_niis_512"

real_nii_path = "/mnt/zhaosheng/4dct/resampled"
real_nii_path_512 = "/mnt/zhaosheng/4dct_data"


npys = sorted([os.path.join(net1_npy_path,_file) for _file in  os.listdir(net1_npy_path) if "real" not in _file and "_t5" in _file])
print(npys)
print(len(npys))
for npy_file in npys:
    
    p_index = npy_file.split("/")[-1].split(".")[0].split("_")[1]
    
   
    input_nii = os.path.join(real_nii_path,f"{p_index}_t0_resampled.nii")
    input_nii_image = ants.image_read(input_nii)
    #input_nii_image.to_file(os.path.join(nii_save_path,f"{epoch}_{p_index}_input.nii.gz"))

    real_nii_512 = os.path.join(real_nii_path_512,f"t0/{p_index}_t0.nii")
    real_512_image = ants.image_read(real_nii_512)
    spacing_512 = real_512_image.spacing
    

    t0_npy = os.path.join(net1_npy_path,f"{net1_epoch}_{p_index}_t0_real.npy")
    t0_array = np.load(t0_npy,allow_pickle=True)[0,0]
    t0_nii_image = input_nii_image.new_image_like(t0_array)
    t0_nii_image.to_file(os.path.join(nii_save_path,f"{p_index}_t0_real.nii.gz"))
    t0_nii_image.to_file(os.path.join(nii_save_path,f"{p_index}_t0_fake.nii.gz"))

    t0_nii_image_512  = ants.resample_image(t0_nii_image,spacing_512,False,4)
    t0_nii_image.to_file(os.path.join(nii_512_save_path,f"{p_index}_t0_real.nii.gz"))
    t0_nii_image.to_file(os.path.join(nii_512_save_path,f"{p_index}_t0_fake.nii.gz"))


    for t in range(1,10):
        t_index = f"t{t}"
        # t_index = npy_file.split("/")[-1].split(".")[0].split("_")[2]
        
        if t<=5:
            real_npy = os.path.join(net1_npy_path,f"{net1_epoch}_{p_index}_{t_index}_real.npy")
            real_npy_array = np.load(real_npy,allow_pickle=True)[0,0]
            real_nii_image = input_nii_image.new_image_like(real_npy_array)
            real_nii_image.to_file(os.path.join(nii_save_path,f"{p_index}_{t_index}_real.nii.gz"))
            real_nii_image_512  = ants.resample_image(real_nii_image,spacing_512,False,4)
            real_nii_image_512.to_file(os.path.join(nii_512_save_path,f"{p_index}_{t_index}_real.nii.gz"))

            fake_npy = os.path.join(net1_npy_path,f"{net1_epoch}_{p_index}_{t_index}.npy")
            npy_array = np.load(fake_npy,allow_pickle=True)[0,0]#*(285+3770)-285 #.transpose(1,2,0)
            fake_nii_image = input_nii_image.new_image_like(npy_array)
            fake_nii_image.to_file(os.path.join(nii_save_path,f"{p_index}_{t_index}_fake.nii.gz"))
            fake_nii_image_512  = ants.resample_image(fake_nii_image,spacing_512,False,4)
            fake_nii_image_512.to_file(os.path.join(nii_512_save_path,f"{p_index}_{t_index}_fake.nii.gz"))
        
        else:
            real_npy = os.path.join(net2_npy_path,f"{net2_epoch}_{p_index}_{t_index}_real.npy")
            real_npy_array = np.load(real_npy,allow_pickle=True)[0,0]
            real_nii_image = input_nii_image.new_image_like(real_npy_array)
            real_nii_image.to_file(os.path.join(nii_save_path,f"{p_index}_{t_index}_real.nii.gz"))
            real_nii_image_512  = ants.resample_image(real_nii_image,spacing_512,False,4)
            real_nii_image_512.to_file(os.path.join(nii_512_save_path,f"{p_index}_{t_index}_real.nii.gz"))
            
            fake_npy = os.path.join(net2_npy_path,f"{net2_epoch}_{p_index}_{t_index}.npy")
            npy_array = np.load(fake_npy,allow_pickle=True)[0,0]#*(285+3770)-285 #.transpose(1,2,0)
            fake_nii_image = input_nii_image.new_image_like(npy_array)
            fake_nii_image.to_file(os.path.join(nii_save_path,f"{p_index}_{t_index}_fake.nii.gz"))
            fake_nii_image_512  = ants.resample_image(fake_nii_image,spacing_512,False,4)
            fake_nii_image_512.to_file(os.path.join(nii_512_save_path,f"{p_index}_{t_index}_fake.nii.gz"))