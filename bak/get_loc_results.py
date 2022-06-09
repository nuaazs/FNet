from antspynet.utilities import lung_extraction
from monai.networks.blocks import Warp
import torch
import ants
import os
import numpy as np
import shutil
from tqdm import tqdm
from utils.log import logger

class Patient():
    def __init__(self,patient_name,imgs_save_path,calc_tumor=False):
        self.warp = Warp()
        self.patient_name = patient_name

        if calc_tumor:
            self.tumor_base = self._getBaseTumor(patient_name)
            print(f"Loading tumor, Shape: {self.tumor_base.shape}")
            self.tumor_base_loc = self._getLoc(self.tumor_base)
            print(f"肿瘤体心 {self.tumor_base_loc}")
        self.imgs_save_path = os.path.join(imgs_save_path,patient_name)
        os.makedirs(self.imgs_save_path,exist_ok=True)

    def _getImage(self,calc_tumor=False):
        patient_name = self.patient_name
        moving = torch.tensor(ants.image_read(f"/home/zhaosheng/4dct_test_nii/{patient_name}/{patient_name}_t0.nii").numpy())
        moving = torch.unsqueeze(moving, dim=0)
        moving = torch.unsqueeze(moving, dim=0)
        print("Get base Image: Success!")

        if calc_tumor:
            real_loc_list = [self.tumor_base_loc]
            fake_loc_list = [self.tumor_base_loc]
        
        for t_index in tqdm(range(1,10)):
            fixed = torch.tensor(ants.image_read(f"/home/zhaosheng/4dct_test_nii/{patient_name}/{patient_name}_t{t_index}.nii").numpy())
            fake_ddf = self._getddf_from_npy(patient_name,t_index,fixed)
            real_ddf = self._getddf_from_nii(patient_name,t_index)
            assert fake_ddf.shape == real_ddf.shape
            fixed = torch.unsqueeze(fixed, dim=0)
            fixed = torch.unsqueeze(fixed, dim=0)
            

            try:
                moved_real = self.warp(moving,real_ddf)
                moved_fake = self.warp(moving,fake_ddf)

                if calc_tumor:
                    tumor_after_real = self.warp(self.tumor_base,real_ddf)
                    tumor_after_fake = self.warp(self.tumor_base,fake_ddf)
            except Exception as e:
                print(e)
                print(f"Err {patient_name} t{t_index}")
                continue


            if calc_tumor:
                real_loc = self._getLoc(tumor_after_real)
                fake_loc = self._getLoc(tumor_after_fake)
                print(f"{t_index} 形变后肿瘤体心 Real:{real_loc}")
                print(f"{t_index} 形变后肿瘤体心 Fake:{fake_loc}")
                real_loc_list.append(real_loc)
                fake_loc_list.append(fake_loc)

            fake_img = ants.from_numpy(moved_fake.numpy()[0,0,:,:,:])
            real_img = ants.from_numpy(fixed.numpy()[0,0,:,:,:])
            input_img = ants.from_numpy(moving.numpy()[0,0,:,:,:])
            ants.image_write(real_img,os.path.join(self.imgs_save_path,f"real_img_{t_index}.nii"))
            ants.image_write(fake_img,os.path.join(self.imgs_save_path,f"fake_img_{t_index}.nii"))
            ants.image_write(input_img,os.path.join(self.imgs_save_path,f"input_img_{t_index}.nii"))
        if calc_tumor:
            return real_loc_list,fake_loc_list
        else:
            return 0,0
    def _getddf_from_nii(self,patient_name,t_index):
        real_ddf = torch.tensor(ants.image_read(f"/home/zhaosheng/4dct_test_nii_transform/{patient_name}_t{t_index}.nii.gz").numpy().transpose(3,0,1,2))
        real_ddf = torch.unsqueeze(real_ddf, dim=0)
        return real_ddf

    def _getddf_from_npy(self,patient_name,t_index,moving_img):
        npy = np.load(f"/home/zhaosheng/paper4/result0502/199_{patient_name}_ddf{t_index}.npy")[0]
        ddf_1 = []
        for index in range(3):
            ddf_raw_1 = ants.from_numpy(npy[index])
            ddf_raw_1 = ants.resample_image(ddf_raw_1,moving_img.shape,True,4)
            ddf_1.append(ddf_raw_1.numpy())
        ddf_1 = torch.tensor(np.array(ddf_1))
        return torch.unsqueeze(ddf_1, dim=0)

    def _getBaseTumor(self,patient_name):
        tumor_base = torch.tensor(ants.image_read(f"/dataset1/4dct_test_tumor/{patient_name}.nrrd").numpy())
        tumor_base = torch.unsqueeze(tumor_base, dim=0)
        tumor_base = torch.unsqueeze(tumor_base, dim=0)
        return tumor_base

    def _getLoc(self,_npy):
        _npy = np.array(_npy)
        loc = np.mean(np.argwhere(_npy == 1),axis=0)
        return loc

if __name__ == "__main__":
    test = Patient("pengmeidi")
    test._getImage()