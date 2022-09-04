import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border, mark_boundaries
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
from glob import glob
from skimage.io import imread
import ants
from PIL import Image


def get_segmented_lungs(raw_im, plot=False):
    '''
    Original function changes input image (ick!)
    '''
    im=raw_im.copy()
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -400
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    
    # return binary
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(15)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone) 
        
    return binary


def get_lung_mask(img):
    """get ct image and return lung mask

    Args:
        nii_file_path (_type_): str

    Returns:
        lung_volume: int, mm3
        lung_mask: numpy.ndarray
        ct: numpy.ndarray
    """
    # print(img.shape)
    array = img.numpy()
    spacing = img.spacing
    shape = img.shape
    slice_num = array.shape[-1]
    mask = []
    for slice in range(slice_num):
        slice_array = array[:,:,slice]
        img = Image.fromarray(slice_array)
        img.save(f"./temp/test{slice}.tif")
        sk_image = imread(f"./temp/test{slice}.tif")
        mask_img = get_segmented_lungs(sk_image,plot=False)
        mask_array = np.array(mask_img)
        mask_array[mask_array>0] = 1

        
        mask.append(mask_array)
    lung_mask = np.array(mask).transpose(1,2,0)
    assert lung_mask.shape == array.shape, "Shape of lung mask and image do not match"
    lung_volume = int(np.sum(lung_mask)*spacing[0]*spacing[1]*spacing[2])
    return {"lung_mask":lung_mask,"lung_volume":lung_volume,"ct":array}


if __name__ == "__main__":
    mask_test = get_lung_mask("/mnt/zhaosheng/FNet/postprocess/niis_from_unet/120829_t5_fake.nii.gz")
    mask_test2 = get_lung_mask("/mnt/zhaosheng/4dct/resampled/120829_t5_resampled.nii")
    print(mask_test["lung_volume"])
    print(mask_test2["lung_volume"])
    # print(mask_test)
