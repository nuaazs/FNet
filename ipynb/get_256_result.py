import ants
import monai
import numpy as np
import matplotlib.pyplot as plt
from time import time
import shutil
import os
from multiprocessing.dummy import Pool as ThreadPool
# paths
SAVE_REAL_256_DDF_PATH = "/pan/4dct_256_real_ddfs"
SAVE_FAKE_256_DDF_PATH = "/pan/4dct_256_fake_ddfs"
SAVE_REAL_RESAMPLED_PATH = "/pan/4dct_256_resampled_real"
SAVE_FAKE_RESAMPLED_PATH = "/pan/4dct_256_resampled_fake"
os.makedirs(SAVE_REAL_256_DDF_PATH,exist_ok=True)
os.makedirs(SAVE_FAKE_256_DDF_PATH,exist_ok=True)
os.makedirs(SAVE_REAL_RESAMPLED_PATH,exist_ok=True)
os.makedirs(SAVE_FAKE_RESAMPLED_PATH,exist_ok=True)
import logging
logging.basicConfig(filename="get_transform_256.log",
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("** Start **")


def get_ddf_and_imgs(input_item):
    t0_256_path,t5_256_path,real_ddf_128_path,fake_ddf_128_path = input_item
    begin_time_all = time()
    pname = t0_256_path.split('/')[-1].split("_")[0]
    tindex = t5_256_path.split('/')[-1].split("_")[1].split(".")[0].replace('t','')
    if os.path.exists(f"{SAVE_FAKE_RESAMPLED_PATH}/{pname}_t{tindex}_fake_fake.nii.gz") and os.path.exists(f"{SAVE_FAKE_RESAMPLED_PATH}/{pname}_t{tindex}_fake_fake.nii.gz"):
        print(f"Pass: {name} t{tindex} already exit!")
        return 0
    try:
        
        logging.info(f"-> Pat Name:{pname} T{tindex}")
        # t0_256_path = "/pan/4dct_niis/t0/355485_t0.nii"
        # t5_256_path = "/pan/4dct_niis/t5/355485_t5.nii"
        # t0_128_path = "/dataset1/4dct_0510/resampled/355485_t0_resampled.nii"
        # t5_128_path = "/dataset1/4dct_0510/resampled/355485_t5_resampled.nii"
        # # ddf_128_path = "/home/zhaosheng/paper4/outputs/A2B/34/ddfs/34_355485_ddf5.npy"
        # ddf_128_path = "/dataset1/4dct_0510/transform/355485_t5_Warp.nii.gz"
        logging.info(f"\t-> Reading imgs ... ")
        t0_256_img = ants.image_read(t0_256_path)
        t5_256_img = ants.image_read(t5_256_path)
        ddf_128_img = ants.image_read(real_ddf_128_path)
        ddf_128_fake_npy = np.load(fake_ddf_128_path)[0].transpose(1,2,3,0)
        ddf_128_fake_img = ddf_128_img.new_image_like(ddf_128_fake_npy)
        logging.info(f"\t-> Doing registeration ... ")
        begin_time = time()
        ddf_256_dict = ants.registration(fixed=t5_256_img,moving=t0_256_img)
        transform_path = ddf_256_dict["fwdtransforms"]

        
        if transform_path:
            logging.info(f"\t-> Saving transform ... ")
            os.makedirs(SAVE_REAL_256_DDF_PATH, exist_ok=True)
            for output_file in transform_path:
                if "nii" in output_file:
                    shutil.move(output_file, f"{SAVE_REAL_256_DDF_PATH}/{pname}_t{tindex}_Warp.nii.gz")

                if "mat" in output_file:
                    shutil.move(output_file, f"{SAVE_REAL_256_DDF_PATH}/{pname}_t{tindex}_GenericAffine.mat")
        else:
            logging.info(f"\t-> Wooooops no transform !!! ")
        end_time = time()
        run_time = end_time-begin_time

        logging.info(f"\t-> Register Done. time used:{run_time}")

        logging.info(f"\t*-> Start resampling ddf ...")
        begin_time = time()
        ddf_256_img = ants.image_read(f"{SAVE_REAL_256_DDF_PATH}/{pname}_t{tindex}_Warp.nii.gz")

        # ========= Real ===========
        spacing_128 = ddf_128_img.spacing
        logging.info(f"\t*-> real before: spacing :{spacing_128}")
        logging.info(f"\t*-> dist spacing :{ddf_256_img.spacing}")

        ddf_128_numpy = ddf_128_img.numpy()
        logging.info(f"\t*-> real before: ddf shape:{ddf_128_numpy.shape}")
        logging.info(f"\t*-> dist ddf shape :{ddf_256_img.shape}")

        new_ddf = []
        for axis in range(3):
            tiny_dff_npy = ddf_128_numpy[...,axis]
            tiny_img = ants.from_numpy(tiny_dff_npy)
            tiny_img.set_spacing(spacing_128)
            tiny_img = ants.resample_image(tiny_img,(ddf_256_img.shape),True,4)
            new_ddf.append(tiny_img.numpy())
        new_ddf = np.array(new_ddf).transpose(1,2,3,0)


        # ========= Fake ===========
        spacing_128_fake = ddf_128_fake_img.spacing
        logging.info(f"\t*-> fake before: spacing :{spacing_128_fake}")
        logging.info(f"\t*-> dist spacing :{ddf_128_fake_img.spacing}")

        ddf_128_numpy_fake = ddf_128_fake_img.numpy()
        logging.info(f"\t*-> fake before: ddf shape:{ddf_128_numpy_fake.shape}")
        logging.info(f"\t*-> dist ddf shape :{ddf_256_img.shape}")

        new_ddf_fake = []
        for axis in range(3):
            tiny_dff_npy = ddf_128_numpy_fake[...,axis]
            tiny_img = ants.from_numpy(tiny_dff_npy)
            tiny_img.set_spacing(spacing_128)
            tiny_img = ants.resample_image(tiny_img,(ddf_256_img.shape),True,4)
            new_ddf_fake.append(tiny_img.numpy())
        new_ddf_fake = np.array(new_ddf_fake).transpose(1,2,3,0)



        
        ddf_256_fake_real_img = ddf_256_img.new_image_like(new_ddf)
        ddf_256_fake_real_img.to_file(f"{SAVE_FAKE_256_DDF_PATH}/{pname}_t{tindex}_Warp_Real.nii.gz")
        logging.info(f"\t*-> new fake-real ddf shape :{ddf_256_fake_real_img.shape}")

        ddf_256_fake_fake_img = ddf_256_img.new_image_like(new_ddf_fake)
        ddf_256_fake_fake_img.to_file(f"{SAVE_FAKE_256_DDF_PATH}/{pname}_t{tindex}_Warp_Fake.nii.gz")
        logging.info(f"\t*-> new fake-fake ddf shape :{ddf_256_fake_real_img.shape}")

        end_time = time()
        run_time = end_time-begin_time
        logging.info(f"\t*-> Resample Done. time used:{run_time}")

        logging.info(f"\t-> Start Warpping ... ")
        real_result = ants.apply_transforms(fixed=t5_256_img,moving=t0_256_img,transformlist=[f"{SAVE_REAL_256_DDF_PATH}/{pname}_t{tindex}_Warp.nii.gz"])
        fake_result_real = ants.apply_transforms(fixed=t5_256_img,moving=t0_256_img,transformlist=[f"{SAVE_FAKE_256_DDF_PATH}/{pname}_t{tindex}_Warp_Real.nii.gz"])
        fake_result_fake = ants.apply_transforms(fixed=t5_256_img,moving=t0_256_img,transformlist=[f"{SAVE_FAKE_256_DDF_PATH}/{pname}_t{tindex}_Warp_Fake.nii.gz"])
        ants.image_write(real_result,f"{SAVE_REAL_RESAMPLED_PATH}/{pname}_t{tindex}.nii.gz")
        ants.image_write(fake_result_real,f"{SAVE_FAKE_RESAMPLED_PATH}/{pname}_t{tindex}_fake_real.nii.gz")
        ants.image_write(fake_result_fake,f"{SAVE_FAKE_RESAMPLED_PATH}/{pname}_t{tindex}_fake_fake.nii.gz")
        total_time = time()-begin_time_all
        logging.info(f"# All Done. Total time used:{total_time}")
    except Exception as e:
        print(e)
        print(f"Error {name} t{tindex}")




if __name__ == "__main__":
    ddf_path = "/home/zhaosheng/paper4/outputs/B2A_0515/34/ddfs"
    name_list = sorted(list(set([_file.split("_")[1] for _file in os.listdir(ddf_path)])))
    print(name_list)
    items = []
    for name in name_list:
        for tindex in range(1,10):
            if tindex<=5:
                ddf_path = "/home/zhaosheng/paper4/outputs/A2B_0515/34/ddfs"
            else:
                ddf_path = "/home/zhaosheng/paper4/outputs/B2A_0515/34/ddfs"
            items.append([f"/dataset1/4dct_0510/resampled/{name}_t0_resampled.nii",
                         f"/dataset1/4dct_0510/resampled/{name}_t{tindex}_resampled.nii",
                         f"/dataset1/4dct_0510/transform/{name}_t{tindex}_Warp.nii.gz",
                         os.path.join(ddf_path,f"34_{name}_ddf{tindex}.npy")])
    print(items[0])
    pool = ThreadPool(4)
    pool.map(get_ddf_and_imgs, items)
    pool.close()
    pool.join()
            # try:
                
            #     get_ddf_and_imgs(t0_256_path = f"/dataset1/4dct_0510/resampled/{name}_t0_resampled.nii",
            #                     t5_256_path = f"/dataset1/4dct_0510/resampled/{name}_t{tindex}_resampled.nii",
            #                     real_ddf_128_path = f"/dataset1/4dct_0510/transform/{name}_t{tindex}_Warp.nii.gz",
            #                     fake_ddf_128_path = os.path.join(ddf_path,f"34_{name}_ddf{tindex}.npy"))
            # except Exception as e:
            #     print(e)
            #     print(f"Error {name} t{tindex}")
