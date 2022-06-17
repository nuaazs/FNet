import ants
import antspynet
import monai
import os
from tqdm import tqdm

def save_largest(raw_img):
    npy = raw_img.numpy()
    npy[npy>0.1]=1
    npy[npy<=0.1]=0
    npy2 = monai.transforms.utils.get_largest_connected_component_mask(npy, connectivity=None)
    post_img = ants.from_numpy(npy2)
    post_img.set_spacing(raw_img.spacing)
    post_img.set_origin(raw_img.origin)
    return post_img

def get_filename(filepath):
    return filepath.split("/")[-1].split(".")[0]
def get_lung_mask(filepath,savepath=None,plot=False):
    lung_path = os.path.join(savepath,get_filename(filepath)+"_lung.nii")
    #right_path = os.path.join(savepath,get_filename(filepath)+"_right_lung.nii")
    if os.path.exists(lung_path):
        return 0,0
    img = ants.image_read(filepath)
    output = antspynet.utilities.lung_extraction(img, modality="ct")
    left_lung = output["probability_images"][1]
    right_lung = output["probability_images"][2]
    left_lung = save_largest(left_lung)
    right_lung = save_largest(right_lung)

    plotpath=os.path.join(savepath,"plots")
    os.makedirs(plotpath,exist_ok=True)
    lung = left_lung+right_lung
    if plot:
        lung.plot(axis=1,overlay=img,overlay_alpha=0.5,filename=os.path.join(plotpath,get_filename(filepath)+".png"))
        #right_lung.plot(axis=1,overlay=img,overlay_alpha=0.5,filename=os.path.join(plotpath,get_filename(filepath)+"_right_lung.png"))
    
    #ants.image_write(left_lung,left_path)
    #ants.image_write(right_lung,right_path)
    ants.image_write(lung,lung_path)
    #print(f"Save to: \n\t{left_path}\n\t{right_path}")
    return left_lung,right_lung

if __name__ == "__main__":
    root = "/zhaosheng_data/4dct_4_test"
    niis = []
    for pname in os.listdir(root):
        niis += [os.path.join(root,pname,_path) for _path in os.listdir(os.path.join(root,pname))]
            #niis += [os.path.join(p_list,_file) for _file in os.listdir(p_list) if ".nii" in _file]

    #niis = [os.path.join(root,_file) for _file in os.listdir(root) if ".nii" in _file]

    for filepath in tqdm(niis):
        # try:
        
        get_lung_mask(filepath,savepath="/dataset1/4dct_4_lungs_test",plot=True)
        # except:
        #     print(f"Pass {filepath}")

