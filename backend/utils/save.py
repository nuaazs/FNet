
# ██╗██╗███╗   ██╗████████╗
# ██║██║████╗  ██║╚══██╔══╝
# ██║██║██╔██╗ ██║   ██║
# ██║██║██║╚██╗██║   ██║
# ██║██║██║ ╚████║   ██║
# ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝
# @Time    : 2022-05-11 09:18:58.000-05:00
# @Author  : 𝕫𝕙𝕒𝕠𝕤𝕙𝕖𝕟𝕘
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : http://iint.icu/
# @File    : /home/zhaosheng/paper4/backend/utils/save.py
# @Describe: Save raw dicom files to disk

import os
import wget

def save_dcm_from_file(file, patient, receive_path):
    """save wav file from post request.

    Args:
        file (request.file): dcm file.
        patient (string): patient id
        receive_path (string): save path

    Returns:
        string: file path
    """
    patient_dir = os.path.join(receive_path, str(patient))
    os.makedirs(patient_dir, exist_ok=True)
    patient_filelist = os.listdir(patient_dir)
    dicom_number = len(patient_filelist) + 1
    # receive wav file and save it to  ->  <receive_path>/<spk_id>/raw_?.webm
    save_path_dcm = os.path.join(patient_dir, f"raw_{dicom_number}.dcm")
    file.save(save_path_dcm)
    return save_path_dcm, dicom_number

def save_dcm_from_url(url, spk, receive_path):
    """save wav file from post request.

    Args:
        file (request.file): dcm file.
        patient (string): patient id
        receive_path (string): save path

    Returns:
        string: file path
    """
    
    patient_dir = os.path.join(receive_path, str(spk))
    os.makedirs(patient_dir, exist_ok=True)
    patient_filelist = os.listdir(patient_dir)
    patient_number = len(patient_filelist) + 1
    # receive wav file and save it to  ->  <receive_path>/<spk_id>/raw_?.dcm
    save_name = f"raw_{patient_number}.dcm"
    if url.startswith("local:"):
        save_path = url.replace("local:", "")
    else:
        save_path = os.path.join(patient_dir, save_name)
        wget.download(url, save_path) # 下载
    return save_path, patient_number