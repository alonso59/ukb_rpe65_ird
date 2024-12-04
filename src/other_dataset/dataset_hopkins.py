import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import eyepy as ep
import cv2
import scipy.io
from PIL import Image
import matplotlib 
from matplotlib import cm
OCT_PATH = 'dataset/0_public_datasets/5_OCT_HOPKINS_CONTROL'

def vol_files(name):
    
    vol_filename = os.path.join(OCT_PATH, name + '.vol')
    del_filename = os.path.join('dataset/0_public_datasets/4_OCT_Manual_Delineations-2018_June_29/delineation/', name + '.mat')
    
    oct_read = ep.import_heyex_vol(vol_filename)
    oct_delin = scipy.io.loadmat(del_filename)
    # print(oct_delin.keys())
    # print(name)
    return oct_read, oct_delin

def get_filenames(path, ext):
    X0 = []
    for i in sorted(os.listdir(path)):
        if i.endswith(ext):
            X0.append(os.path.join(path, i))
    return X0

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_mask(name, oct_read, oct_delin, bscanidx, pathimg, pathmsk, pathimgsl, pathmsksl):
    data = oct_read[bscanidx].data
    delinations = []
    for i in range(10):
        layer = oct_delin['control_pts'][bscanidx][i]
        # print(layer.shape)
        try:
            layer = layer[layer[:, 0].argsort()]
        
            f = interpolate.interp1d(layer[:,0], layer[:,1])
            array = np.zeros(data.shape[1])
            x = np.arange(layer.min(), layer.max(), 1)
            y = f(x)
            offset = 0
            if i == 6 or i == 7 or i == 8:
                offset = 2
            y = y-offset
            k = 0
            for n in range(layer.min().astype('uint32'), layer.max().astype('uint32'), 1):
                array[n] = y[k]
                k += 1
            delinations.append(array)
        except:
            continue
    norm = matplotlib.colors.Normalize(vmin=0, vmax=8)
    mask = np.zeros((data.shape[0], data.shape[1])).astype('uint8')
    delinations = np.array(delinations).astype('uint16')

    for i in range(data.shape[1]):
    #     # RNFL 1
    #     mask[delinations[0,i]:delinations[1,i], i, :] = [cm.hsv(norm(1), bytes=True)[0], cm.hsv(norm(1), bytes=True)[1], cm.hsv(norm(1), bytes=True)[2]]  if delinations[0,i] <= delinations[1,i] and delinations[0,i] > 0 and delinations[1,i] > 0 else mask[delinations[0,i]:delinations[1,i], i]
    #     # GCL-IPL 2
    #     mask[delinations[1,i]:delinations[2,i], i, :] = [cm.hsv(norm(2), bytes=True)[0], cm.hsv(norm(2), bytes=True)[1], cm.hsv(norm(2), bytes=True)[2]]   if delinations[1,i] <= delinations[2,i] and delinations[1,i] > 0 and delinations[2,i] > 0 else mask[delinations[1,i]:delinations[2,i], i]
    #     # INL 3
    #     mask[delinations[2,i]:delinations[3,i], i, :] = [cm.hsv(norm(3), bytes=True)[0], cm.hsv(norm(3), bytes=True)[1], cm.hsv(norm(3), bytes=True)[2]]   if delinations[2,i] <= delinations[3,i] and delinations[2,i] > 0 and delinations[3,i] > 0 else mask[delinations[2,i]:delinations[3,i], i]
    #     # OPL 4
    #     mask[delinations[3,i]:delinations[4,i], i, :] = [cm.hsv(norm(4), bytes=True)[0], cm.hsv(norm(4), bytes=True)[1], cm.hsv(norm(4), bytes=True)[2]]   if delinations[3,i] <= delinations[4,i] and delinations[3,i] > 0 and delinations[4,i] > 0 else mask[delinations[3,i]:delinations[4,i], i]
    #     # ONL 5
    #     mask[delinations[4,i]:delinations[5,i], i, :] = [cm.hsv(norm(5), bytes=True)[0], cm.hsv(norm(5), bytes=True)[1], cm.hsv(norm(5), bytes=True)[2]]   if delinations[4,i] <= delinations[5,i] and delinations[4,i] > 0 and delinations[5,i] > 0 else mask[delinations[4,i]:delinations[5,i], i]
    #     # IS 6
    #     mask[delinations[5,i]:delinations[6,i], i, :] = [cm.hsv(norm(6), bytes=True)[0], cm.hsv(norm(6), bytes=True)[1], cm.hsv(norm(6), bytes=True)[2]] if delinations[5,i] <= delinations[6,i] and delinations[5,i] > 0 and delinations[6,i] > 0 else mask[delinations[5,i]:delinations[6,i], i]
    #     # OS 7
    #     mask[delinations[6,i]:delinations[7,i], i, :] = [cm.hsv(norm(7), bytes=True)[0], cm.hsv(norm(7), bytes=True)[1], cm.hsv(norm(7), bytes=True)[2]]   if delinations[6,i] <= delinations[7,i] and delinations[6,i] > 0 and delinations[7,i] > 0 else mask[delinations[6,i]:delinations[7,i], i]
    #     # RPE 8
    #     mask[delinations[7,i]:delinations[8,i], i, :] = [cm.hsv(norm(8), bytes=True)[0], cm.hsv(norm(8), bytes=True)[1], cm.hsv(norm(8), bytes=True)[2]]   if delinations[7,i] <= delinations[8,i] and delinations[7,i] > 0 and delinations[8,i] > 0 else mask[delinations[7,i]:delinations[8,i], i]
        # RNFL 1
        # mask[delinations[0,i]:delinations[1,i], i] = 1  if delinations[0,i] <= delinations[1,i] and delinations[0,i] > 0 and delinations[1,i] > 0 else mask[delinations[0,i]:delinations[1,i], i]
        # # GCL-IPL 2                                  
        # mask[delinations[1,i]:delinations[2,i], i] = 2  if delinations[1,i] <= delinations[2,i] and delinations[1,i] > 0 and delinations[2,i] > 0 else mask[delinations[1,i]:delinations[2,i], i]
        # # INL 3                                      
        # mask[delinations[2,i]:delinations[3,i], i] = 3  if delinations[2,i] <= delinations[3,i] and delinations[2,i] > 0 and delinations[3,i] > 0 else mask[delinations[2,i]:delinations[3,i], i]
        # # OPL 4                                      
        # mask[delinations[3,i]:delinations[4,i], i] = 4  if delinations[3,i] <= delinations[4,i] and delinations[3,i] > 0 and delinations[4,i] > 0 else mask[delinations[3,i]:delinations[4,i], i]
        # # ONL 5                                      
        # mask[delinations[4,i]:delinations[5,i], i] = 5  if delinations[4,i] <= delinations[5,i] and delinations[4,i] > 0 and delinations[5,i] > 0 else mask[delinations[4,i]:delinations[5,i], i]
        # # IS 6                                       
        # mask[delinations[5,i]:delinations[6,i], i] = 6  if delinations[5,i] <= delinations[6,i] and delinations[5,i] > 0 and delinations[6,i] > 0 else mask[delinations[5,i]:delinations[6,i], i]
        # # OS 7                                       
        # mask[delinations[6,i]:delinations[7,i], i] = 7  if delinations[6,i] <= delinations[7,i] and delinations[6,i] > 0 and delinations[7,i] > 0 else mask[delinations[6,i]:delinations[7,i], i]
        # # RPE 8                                      
        # mask[delinations[7,i]:delinations[8,i], i] = 8  if delinations[7,i] <= delinations[8,i] and delinations[7,i] > 0 and delinations[8,i] > 0 else mask[delinations[7,i]:delinations[8,i], i]
        # OPL 2
        mask[delinations[0,i]:delinations[1,i], i] = 2 if delinations[0,i] <= delinations[1,i] and delinations[0,i] > 0 and delinations[1,i] > 0 else mask[delinations[0,i]:delinations[1,i], i]
        # ELM 3
        mask[delinations[2,i]:delinations[3,i], i] = 3 if delinations[2,i] <= delinations[3,i] and delinations[2,i] > 0 and delinations[3,i] > 0 else mask[delinations[2,i]:delinations[3,i], i]
        # EZ 1
        mask[delinations[3,i]:delinations[4,i], i] = 1 if delinations[3,i] <= delinations[4,i] and delinations[3,i] > 0 and delinations[4,i] > 0 else mask[delinations[3,i]:delinations[4,i], i]
        # BM 4
        mask[delinations[4,i]:delinations[5,i], i] = 4 if delinations[4,i] <= delinations[5,i] and delinations[4,i] > 0 and delinations[5,i] > 0 else mask[delinations[4,i]:delinations[5,i], i]

    img = Image.fromarray(data)
    msk = Image.fromarray(mask)
    # img = img.convert("RGBA")
    # msk = msk.convert("RGBA")
    # msk = Image.blend(img, msk, 0.4)
    img.save(pathimg + name + f"_{bscanidx}.png")
    msk.save(pathmsk + name + f"_{bscanidx}.png")
    slicing(name, data, mask, pathimgsl, pathmsksl, bscanidx, size=128, shift=64)


def get_dataset(name, pathimg, pathmsk, pathimgsl, pathmsksl):
    oct_read, oct_delin = vol_files(name)
    for bscanidx in range(0, 49):
        get_mask(name, oct_read, oct_delin, bscanidx, pathimg, pathmsk, pathimgsl, pathmsksl)



def slicing(name, image, mask, path_img, path_msk, bscanidx, size=128, shift=128):
    j = 1
    for i in range((size), (image.shape[1]), shift):
        img_save = image[:, i - size:i]
        msk_save = mask[:, i - size:i]
        img = Image.fromarray(img_save)
        msk = Image.fromarray(msk_save)
        img.save(path_img + name + f"_{bscanidx}" + f"_{j}.png")
        msk.save(path_msk + name + f"_{bscanidx}" + f"_{j}.png")
        j += 1

def main():
    base_path = 'dataset2/hms_control_5c_v2/'
    train_path_images = base_path + 'train/Images/'
    train_path_masks = base_path + 'train/Masks/'
    val_path_images = base_path + 'val/Images/'
    val_path_masks = base_path + 'val/Masks/'
    slices_images_train = base_path + 'train/images_slices/'
    slices_masks_train = base_path + 'train/masks_slices/'
    slices_images_val = base_path + 'val/images_slices/'
    slices_masks_val = base_path + 'val/masks_slices/'

    create_dir(train_path_images)
    create_dir(train_path_masks)
    create_dir(val_path_images)
    create_dir(val_path_masks)
    create_dir(slices_images_train)
    create_dir(slices_masks_train)
    create_dir(slices_images_val)
    create_dir(slices_masks_val)

    filenames_oct = get_filenames(OCT_PATH, 'vol')

    train, val = train_test_split(filenames_oct, train_size=0.8, shuffle=True)

    for t in tqdm(train):
        try:
            name = os.path.splitext(os.path.split(t)[1])[0]
            get_dataset(name, train_path_images, train_path_masks, slices_images_train, slices_masks_train)
        except:
            continue
    for v in tqdm(val):
        try:
            name = os.path.splitext(os.path.split(v)[1])[0]
            get_dataset(name, val_path_images, val_path_masks, slices_images_val, slices_masks_val)
        except:
            continue


if __name__ == "__main__":
    main()
