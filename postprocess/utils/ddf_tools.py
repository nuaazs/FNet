import numpy as np
import ants


def load_npy_ddf(npy_array_filepath,real_ddf_filepath,savepath="./temp.nii.gz"):
    real_ddf_img = ants.image_read(real_ddf_filepath)
    
    npy_array = np.load(npy_array_filepath,allow_pickle=True)[0].transpose(1,2,3,0)
    ddf_img = real_ddf_img.new_image_like(npy_array)
    ddf_img.to_file(savepath)
    # # print(npy_array[0].shape)
    # # print(shape)
    # # if npy_array[0].shape != shape:
    # #     resample_3d_npy(npy_array,shape)
    # ddf_img = ants.from_numpy(npy_array,is_rgb=True)
    # ants.set_spacing(ddf_img,(spacing[0],spacing[1],spacing[2]))
    # ants.image_write(ddf_img,savepath)
    # print(f"ddf fake img , shape:{ddf_img.shape} spacing:{ddf_img.spacing}")
    return ddf_img,savepath
    
def resample_3d_ddf(ddf,img):
    print(f" DDF shape: {ddf.shape}")
    print(f" IMG shape: {img.shape}")

    dist_shape = list(img.shape)
    print(f" dist_shape: {dist_shape}")
    ddf_result = []
    for i in range(3):
        now_ddf = ddf[i]
        fake_ddf_image_256 = ants.from_numpy(now_ddf)
        fake_ddf_image_512  = ants.resample_image(fake_ddf_image_256,dist_shape,True,4)
        
        ddf_result.append(fake_ddf_image_512.numpy()*4)    
    return np.array(ddf_result)


def warp_img_npy(fixed_filepath,moving_filepath,ddf_npy_filepath,real_ddf_filepath):
    fixed = ants.image_read(fixed_filepath)
    moving = ants.image_read(moving_filepath)
    ddf_img,savepath = load_npy_ddf(ddf_npy_filepath,real_ddf_filepath)
    moved_img = ants.apply_transforms( fixed=fixed, moving=moving,transformlist=[savepath])
    result = {
        "fixed":fixed,
        "moving":moving,
        "ddf":ddf_img,
        "moved":moved_img
    }
    return result

def warp_img_nii(fixed_filepath,moving_filepath,ddf_nii_filepath):
    fixed = ants.image_read(fixed_filepath)
    moving = ants.image_read(moving_filepath)
    ddf_img = ants.image_read(ddf_nii_filepath)
    if ddf_img.shape != moving.shape:
        ddf_npy = ddf_img.numpy().transpose(3,0,1,2)
        ddf_npy_reshaped = resample_3d_npy(ddf_npy,moving.shape)
        ddf_img_reshaped = ants.from_numpy(ddf_npy_reshaped)
        ants.set_spacing(ddf_img_reshaped,(moving.spacing[0],moving.spacing[1],moving.spacing[2]))
        ants.image_write(ddf_img_reshaped,"./temp_real_ddf.nii")
        moved_img = ants.apply_transforms( fixed=fixed, moving=moving,transformlist=["./temp_real_ddf.nii"])
    else:
        moved_img = ants.apply_transforms( fixed=fixed, moving=moving,transformlist=[ddf_nii_filepath])
    result = {
        "fixed":fixed,
        "moving":moving,
        "ddf":ddf_img,
        "moved":moved_img
    }
    return result

def resample_3d_npy(npy_array,shape):
    ddf_1 = []
    for index in range(3):
        ddf_npy_1 = npy_array[0,index]
        ddf_raw_1 = ants.from_numpy(ddf_npy_1)
        ddf_raw_1 = ants.resample_image(ddf_raw_1,shape,True,4)
        ddf_1.append(ddf_raw_1.numpy())
    ddf_1 = np.array(ddf_1)
    return ddf_1
