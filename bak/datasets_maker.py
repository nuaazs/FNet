# ██╗██╗███╗   ██╗████████╗
# ██║██║████╗  ██║╚══██╔══╝
# ██║██║██╔██╗ ██║   ██║
# ██║██║██║╚██╗██║   ██║
# ██║██║██║ ╚████║   ██║
# ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝

# @Time    : 2021-09-28 09:35:49
# @Author  : zhaosheng
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : iint.icu
# @File    : datasets_maker.py
# @Describe: generate datases from raw .dcm files (mri-ct project)

import os
import re
import pydicom
import shutil
import numpy as np
from PIL import Image
import SimpleITK as sitk
import nibabel as nib
import ants
from time import *

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "80"
os.environ["ANTS_RANDOM_SEED"] = "3"

class DatasetsMaker(object):
    """Generate 4DCT dataset

    Args:
        object (_type_): _description_
    """
    
    def __init__(self,root_path=r"/zhaosheng_data/4dct/data",nii_path=f"/zhaosheng_data/4dct/niis/",parten="\d{6}"):
        """init function

        Args:
            root_path (str, optional): path to raw data directory. Defaults to r"/zhaosheng_data/4dct/data".
            nii_path (str, optional): path to save niis. Defaults to f"/zhaosheng_data/4dct/niis/".
            parten (regexp, optional): fold filter. Defaults to "\d{6}".
        """
        self.root_path = root_path
        self._change_dir(root_path)
        _, dirs, _ = next(os.walk(root_path))
        self.dirs = [dir_name for dir_name in dirs if re.findall(parten, dir_name)!=[]]
        self.nii_path = nii_path
        self.transform_path = os.path.join(self.nii_path,"transform")
        os.makedirs(self.transform_path,exist_ok=True)
        
    def _clasify(self):
        """clasify different dicom by seriesName, and save them to different folds.

        Returns:
            bool: success or not
        """
        for patient_id,dir_name in enumerate(self.dirs,1):
            print(dir_name)
            fold = os.path.join(self.root_path,dir_name)
            dicoms = [os.path.join(fold,dicom_name) for dicom_name in os.listdir(fold) if ".dcm" in dicom_name]
            for file_path in dicoms:
                try:
                    filename = file_path.split("\\")[-1]
                    dcm = pydicom.read_file(file_path)
                    seriesUid,seriesName = dcm.SeriesInstanceUID,dcm.SeriesDescription
                    save_path = fold+"/"+seriesName+"/"
                    os.makedirs(save_path, exist_ok=True)
                    shutil.move(file_path,save_path)
                except:
                    pass
            return True
    
    def _normalization(self,data_inp):
        """normalization numpy array from

        Args:
            data_inp (nparray): input numpy array

        Returns:
            nparray: normalized array.
        """
        data = data_inp.copy()
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / (_range)

    def _generate_nii(self,save_numpy=True):
        """generate nii from mri_dicoms

        Args:
            save_numpy (bool, optional): save numpy or not. Defaults to True.

        Returns:
            None
        """
        for patient_id,dir_name in enumerate(self.dirs,1):
            patient_id = dir_name.replace(" ","")
            fold = os.path.join(self.root_path,dir_name)
            _,dirs, _ = next(os.walk(fold))
            dirs = [dir_name for dir_name in dirs]
            print(f"\t-> {patient_id}：")
            for mri_dir in [_dirs for _dirs in dirs if "Reg" not in _dirs and "ipynb" not in _dirs and "Set" not in _dirs and "Doses" not in _dirs]:
                mri_path = os.path.join(fold,mri_dir)
                mode = mri_dir.split("\\")[-1]
                data_mode = "output"
                if " 0% " in mode:
                    mode = str(patient_id)+"_t0"
                    data_mode = "input"
                elif " 10% " in mode:
                    mode = str(patient_id)+"_t1"
                elif " 20% " in mode:
                    mode = str(patient_id)+"_t2"
                elif " 30% " in mode:
                    mode = str(patient_id)+"_t3"
                elif " 40% " in mode:
                    mode = str(patient_id)+"_t4"
                elif " 50% " in mode:
                    data_mode = "input"
                    mode = str(patient_id)+"_t5"
                elif " 60% " in mode:
                    mode = str(patient_id)+"_t6"
                elif " 70% " in mode:
                    mode = str(patient_id)+"_t7"
                elif " 80% " in mode:
                    mode = str(patient_id)+"_t8"
                elif " 90% " in mode:
                    mode = str(patient_id)+"_t9"
                else:
                    continue
                print(f"\t\t->{mode}")
                os.makedirs(self.nii_path,exist_ok=True)
                image_path = os.path.join(self.nii_path,f"{mode}.nii")
                
                mri_reader = sitk.ImageSeriesReader()
                mri_dicoms = mri_reader.GetGDCMSeriesFileNames(mri_path)
                mri_reader.SetFileNames(mri_dicoms)
                mri_img = mri_reader.Execute()
                mri_size = mri_img.GetSize()
                mri_img = sitk.Cast(mri_img, sitk.sitkFloat32)
                sitk.WriteImage(mri_img,image_path)
                
        return 0
    
    
    def _renamedir(self,save_numpy=True):
        """ !

        Args:
            save_numpy (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        for patient_id,dir_name in enumerate(self.dirs,1):
            patient_id = str(patient_id).zfill(3)
            fold = os.path.join(self.root_path,dir_name)
            _,dirs, _ = next(os.walk(fold))
            dirs = [dir_name for dir_name in dirs]
            print(f"\t-> {patient_id}：")
            for mri_dir in [_dirs for _dirs in dirs if "Reg" not in _dirs and "ipynb" not in _dirs and "Set" not in _dirs and "npy" not in _dirs and "Doses" not in _dirs]:
                mri_path = os.path.join(fold,mri_dir)
                mode = mri_dir.split("\\")[-1]
                data_mode = "output"
                if " 0% " in mode:
                    mode = str(patient_id)+"_t0"
                elif " 10% " in mode:
                    mode = str(patient_id)+"_t1"
                elif " 20% " in mode:
                    mode = str(patient_id)+"_t2"
                elif " 30% " in mode:
                    mode = str(patient_id)+"_t3"
                elif " 40% " in mode:
                    mode = str(patient_id)+"_t4"
                elif " 50% " in mode:
                    mode = str(patient_id)+"_t5"
                elif " 60% " in mode:
                    mode = str(patient_id)+"_t6"
                elif " 70% " in mode:
                    mode = str(patient_id)+"_t7"
                elif " 80% " in mode:
                    mode = str(patient_id)+"_t8"
                elif " 90% " in mode:
                    mode = str(patient_id)+"_t9"
                else:
                    continue
                print(f"Change dir:{mri_dir}\n\t->/{fold}/{mode}/")
                #shutil.move(mri_dir,f"/zhaosheng_data/CT4D/temp/{mode}_t1/")
                #print(f"\t\t->{mode}")
        return 0


    
    def _resize_image_itk(self,itkimage, newSpacing, originSpcaing, resamplemethod=sitk.sitkNearestNeighbor):
        """resize dicom by itk

        Args:
            itkimage (_type_): _description_
            newSpacing (_type_): _description_
            originSpcaing (_type_): _description_
            resamplemethod (_type_, optional): _description_. Defaults to sitk.sitkNearestNeighbor.

        Returns:
            _type_: _description_
        """
        newSpacing = np.array(newSpacing, float)
        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()
        factor = newSpacing / originSpcaing
        newSize = originSize / factor
        newSize = newSize.astype(np.int)
        resampler.SetReferenceImage(itkimage)  # Set the size, origin, spacing, and orientation of the output to itkimage
        resampler.SetOutputSpacing(newSpacing.tolist())  # Set the output image spacing
        resampler.SetSize(newSize.tolist())  # Set the size of the output image
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(itkimage)
        return itkimgResampled

    def _DownsamplingDicomFixedResolution(self,heightspacing_new,widthspacing_new,img):
        """Downsampling dicom by ITK

        Args:
            heightspacing_new (_type_): _description_
            widthspacing_new (_type_): _description_
            img (_type_): _description_

        Returns:
            _type_: _description_
        """
        Spacing = img.GetSpacing()
        print(f"\t\tSpacing before : {Spacing}")
        thickspacing, widthspacing, heightspacing = Spacing[2], Spacing[0], Spacing[1]
        resampled_img = self._resize_image_itk(img, newSpacing=(widthspacing_new, heightspacing_new, thickspacing),
                                originSpcaing=(widthspacing, heightspacing, thickspacing),
                                resamplemethod=sitk.sitkNearestNeighbor)
        print(f"\t\tSpacing after : {resampled_img.GetSpacing()}")
        return resampled_img

    def _change_dir(self,root_path):
        """change dir to path

        Args:
            root_path (str): dest root_path

        Returns:
            None
        """
        os.chdir(self.root_path)
        return 0

    @staticmethod
    def get_transform(moving_file, fixed_file,type_of_transform,plots_path,transform_path,resampled_path,fixed_shape=(128,128,64)):
        """get transform between two antsImages and save reshaped image."

        Args:
            moving_file (antsImage): moving Image t0
            fixed_file (antsImage): fixed Image tn
            type_of_transform (str): type_of_tr
            plots_path (str): where to save the pngs
            transform_path (str): where to save the transforms
            resampled_path (str): where to save the resampled images
            fixed_shape (tuple, optional): image fixed_shape. Defaults to (128,128,64).

        Returns:
            None
        """
        begin_time = time()
        filename = fixed_file.split("/")[-1].split(".")[-2]
        print(f"\n=> Now loading:{filename}")
        moving = ants.image_read(moving_file)
        fixed = ants.image_read(fixed_file)

        # Resample image        
        if fixed_shape:
            moving_resampled = ants.resample_image(moving,fixed_shape,True,4)
            fixed_resampled = ants.resample_image(fixed,fixed_shape,True,4)
        else:
            new_spacing = np.array(moving.spacing)*4
            moving_resampled = ants.resample_image(moving,new_spacing,False,4)
            fixed_resampled = ants.resample_image(fixed,new_spacing,False,4)

        
        if resampled_path:
            os.makedirs(resampled_path, exist_ok=True)
            ants.image_write(fixed_resampled, os.path.join(resampled_path, filename.split("_")[0]+"_t0_resampled.nii"))
            ants.image_write(moving_resampled, os.path.join(resampled_path, filename+"_resampled.nii"))

        
 
        print(f"\t-> Registration ...")
        reg = ants.registration(
            fixed=fixed_resampled, moving=moving_resampled, type_of_transform=type_of_transform)
        moved = reg["warpedmovout"]
        print(f"\t-> Plot ...")
        if plots_path:
            os.makedirs(plots_path,exist_ok=True)
            moved_plot = moved + 1000
            fixed_resampled_plot = fixed_resampled + 1000
            moved_plot.plot(title='moved', axis=1, cbar=True,
                filename=os.path.join(plots_path, filename+"_moved.png"))
            ants.plot(moved_plot, overlay=fixed_resampled_plot, overlay_cmap='hot', overlay_alpha=0.5,
                    axis=1, cbar=True, filename=os.path.join(plots_path, filename+"_overlay_1.png"))
            ants.plot(moved_plot, overlay=fixed_resampled_plot, overlay_cmap='hot', overlay_alpha=0.5,
                    axis=0, cbar=True, filename=os.path.join(plots_path, filename+"_overlay_0.png"))

        
        
        print(f"\t-> Save transforms ...")
        if transform_path:
            os.makedirs(transform_path, exist_ok=True)
            for output_file in reg["fwdtransforms"]:
                if "nii" in output_file:
                    shutil.move(output_file, f"{transform_path}/{filename}_Warp.nii.gz")
                if "mat" in output_file:
                    shutil.move(output_file, f"{transform_path}/{filename}_GenericAffine.mat")
        end_time = time()
        run_time = end_time-begin_time
        print(f"\t-> Used:{run_time}s")