import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import eyepy
from eyepy.core.base import Oct, LayerAnnotation
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from utils import get_filenames, create_dir
import matplotlib 
from matplotlib import cm
OCT_PATH = 'dataset/2_OCTAnnotated'

def vol_files(name):
    vol_filename = os.path.join(OCT_PATH, name + '.vol')
    oct_read = Oct.from_heyex_vol(vol_filename)
    # print(oct_read.meta)
    
    # print(oct_read.bscans[0].annotation["layers"]['OPL'])
    # sys.exit()
    return oct_read

def get_annotations(oct_read):
    ILM = np.round(
        oct_read.bscans[0].annotation["layers"]['ILM']).astype('uint16')
    try:
        RNFL = np.round(
            oct_read.bscans[0].annotation["layers"]['RNFL']).astype('uint16')
    except:
        RNFL = np.zeros((ILM.shape[0])).astype('uint16')
        pass
    try:
        GCL = np.round(
            oct_read.bscans[0].annotation["layers"]['GCL']).astype('uint16')
    except:
        GCL = np.zeros((ILM.shape[0])).astype('uint16')
        pass
    try:
        IPL = np.round(
            oct_read.bscans[0].annotation["layers"]['IPL']).astype('uint16')
    except:
        IPL = np.zeros((ILM.shape[0])).astype('uint16')
        pass
    # ## OPL ##
    OPL = np.round(
        oct_read.bscans[0].annotation["layers"]['OPL']).astype('uint16')
    INL = np.round(
        oct_read.bscans[0].annotation["layers"]['INL']).astype('uint16')
    # ## EZ ##
    PR2 = np.round(
        oct_read.bscans[0].annotation["layers"]['PR2']).astype('uint16')
    PR1 = np.round(
        oct_read.bscans[0].annotation["layers"]['PR1']).astype('uint16')
    # ## BM ##
    try:
        BM = np.round(
            oct_read.bscans[0].annotation["layers"]['BM']).astype('uint16')
    except:
        BM = np.zeros((PR1.shape[0])).astype('uint16')
        pass
    # ## ELM ##
    try:
        ELM = np.round(
            oct_read.bscans[0].annotation["layers"]['ELM']).astype('uint16')
    except:
        ELM = np.zeros((PR1.shape[0])).astype('uint16')
        pass
    return OPL, INL, PR2, PR1, BM, ELM, ILM, RNFL, GCL, IPL


def get_images_masks(file):
    name = os.path.splitext(os.path.split(file)[1])[0]

    oct_read = vol_files(name)

    data = oct_read.bscans[0].scan
    data = np.expand_dims(data, axis=-1)

    zeros = np.zeros((data.shape[0], data.shape[1], 3)).astype('uint8')
    annotation = np.add(data, zeros)
    OPL, INL, PR2, PR1, BM, ELM, ILM, RNFL, GCL, IPL = get_annotations(oct_read)

    # Generate ground truth
    mask = np.zeros((data.shape[0], data.shape[1])).astype('uint8')

    norm = matplotlib.colors.Normalize(vmin=0, vmax=4)

    for i in range(OPL.shape[0]):
        annotation[INL[i], i, :] = [cm.hsv(norm(0), bytes=True)[0], cm.hsv(norm(2), bytes=True)[1], cm.hsv(norm(2), bytes=True)[2]]
        annotation[OPL[i], i, :] = [cm.hsv(norm(10), bytes=True)[0], cm.hsv(norm(2), bytes=True)[1], cm.hsv(norm(2), bytes=True)[2]]
        annotation[ELM[i], i, :] = [cm.hsv(norm(20), bytes=True)[0], cm.hsv(norm(3), bytes=True)[1], cm.hsv(norm(3), bytes=True)[2]]
        annotation[PR1[i], i, :] = [cm.hsv(norm(30), bytes=True)[0], cm.hsv(norm(1), bytes=True)[1], cm.hsv(norm(1), bytes=True)[2]]
        annotation[PR2[i], i, :] = [cm.hsv(norm(40), bytes=True)[0], cm.hsv(norm(1), bytes=True)[1], cm.hsv(norm(1), bytes=True)[2]]
        annotation[BM[i], i, :]  = [cm.hsv(norm(49), bytes=True)[0], cm.hsv(norm(4), bytes=True)[1], cm.hsv(norm(4), bytes=True)[2]]
        
        annotation[RNFL[i], i, :] = [100, 200, 50]
        annotation[GCL[i], i, :] = [100, 100, 255]  
        annotation[ILM[i], i, :] = [255, 100, 100]  
        annotation[INL[i], i,  :] = [50, 50, 255]
        annotation[OPL[i], i,  :] = [0, 255, 255]
        annotation[ELM[i], i, :] = [150, 100, 255]
        annotation[PR1[i], i, :] = [0, 255, 50]
        annotation[PR2[i], i, :] = [0, 255, 50]
        annotation[BM[i], i, :] = [255, 0, 0]
        
        # annotation[IPL[i], i, :] = [255, 255, 0]
        # OPL
        mask[INL[i]:OPL[i], i] = 2 if INL[i] <= OPL[i] and INL[i] > 0 and OPL[i] > 0 else mask[INL[i]:OPL[i], i]
        # ELM
        mask[ELM[i]:PR1[i], i] = 3 if ELM[i] <= PR1[i] and ELM[i] > 0 and PR1[i] > 0 else mask[ELM[i]:PR1[i], i]
        # EZ
        mask[PR1[i]:PR2[i], i] = 1 if PR1[i] <= PR2[i] and PR1[i] > 0 and PR2[i] > 0 else mask[PR1[i]:PR2[i], i]
        # IZ-RPE
        mask[PR2[i]:BM[i], i] = 4 if PR2[i] <= BM[i] and PR2[i] > 0 and BM[i] > 0 else mask[PR2[i]:BM[i], i]
        # # OPL
        # mask[INL[i]:OPL[i], i, :] = [cm.hsv(norm(2), bytes=True)[0], cm.hsv(norm(2), bytes=True)[1], cm.hsv(norm(2), bytes=True)[2]] if INL[i] <= OPL[i] and INL[i] > 0 and OPL[i] > 0 else mask[INL[i]:OPL[i], i, :]
        # # ELM
        # mask[ELM[i]:PR1[i], i, :] = [cm.hsv(norm(3), bytes=True)[0], cm.hsv(norm(3), bytes=True)[1], cm.hsv(norm(3), bytes=True)[2]] if ELM[i] <= PR1[i] and ELM[i] > 0 and PR1[i] > 0 else mask[ELM[i]:PR1[i], i, :]
        # # EZ
        # mask[PR1[i]:PR2[i], i, :] = [cm.hsv(norm(1), bytes=True)[0], cm.hsv(norm(1), bytes=True)[1], cm.hsv(norm(1), bytes=True)[2]] if PR1[i] <= PR2[i] and PR1[i] > 0 and PR2[i] > 0 else mask[PR1[i]:PR2[i], i, :]
        # # IZ-RPE
        # mask[PR2[i]:BM[i], i, :] = [cm.hsv(norm(4), bytes=True)[0], cm.hsv(norm(4), bytes=True)[1], cm.hsv(norm(4), bytes=True)[2]] if PR2[i] <= BM[i] and PR2[i] > 0 and BM[i] > 0 else mask[PR2[i]:BM[i], i, :]
    
    return mask, annotation


def crop_overlap(oct, name, image, mask, path_img, path_msk, size=128, shift=64):
    """
    file: OCT file extention .vol
    image: Numpy array
    mask: Numpy array
    path_img: path to save patches
    path_msk: path to save patches
    size: image size
    """
    OPL, INL, PR2, PR1, BM, ELM, ILM, RNFL, GCL, IPL  = get_annotations(oct)
    j = 1
    k = 0
    for i in range((size), (image.shape[1]), shift):
        # min_pixel = np.max(INL[k:i])
        # max_pixel = np.max(PR2[k:i])
        min_pixel = np.max(ILM[k:i])
        max_pixel = np.max(BM[k:i])
        if min_pixel != 0 and max_pixel != 0 and max_pixel > min_pixel:
            delta1 = max_pixel - min_pixel
            delta2 = size - delta1
            delta3 = delta2 // 2
            delta4 = min_pixel - delta3
            delta5 = max_pixel + delta3
            if delta2 % 2 != 0:
                delta5 += 1
            if delta4 < 0:
                delta4 = 0
                delta5 = size
            if delta5 > image.shape[0]:
                delta5 = image.shape[0]
                delta4 = delta5 - size
            img_save = image[delta4:delta5, i - size:i]
            msk_save = mask[delta4:delta5, i - size:i]
            img = Image.fromarray(img_save)
            msk = Image.fromarray(msk_save)
            # img.save(path_img + '_' + name + f"_{j}.png")
            # msk.save(path_msk + '_' + name + f"_{j}.png")
            j += 1
        k = i

def slicing(image, mask, path_img, name, path_msk, size=128, shift=64):
    j = 1
    for i in range((size), (image.shape[1]), shift):
        img_save = image[:, i - size:i]
        msk_save = mask[:, i - size:i]
        img = Image.fromarray(img_save)
        msk = Image.fromarray(msk_save)
        img.save(path_img + '_' + name + f"_{j}.png")
        msk.save(path_msk + '_' + name + f"_{j}.png")
        j += 1

def do_dataset(file, data, name, path_img_save, path_msk_save, path_ann_save):
    mask, annotations = get_images_masks(file)
    img = Image.fromarray(data)
    msk = Image.fromarray(mask)
    ann = Image.fromarray(annotations)
    img.save(path_img_save + name + f".png")
    msk.save(path_msk_save + name + f".png")
    ann.save(path_ann_save + name + f".png")
    return mask


def main():
    base_path = 'dataset1/rand_corrected_13_03_v3/'

    train_path_images = base_path + 'train/Images/'
    train_path_ipatches = base_path + 'train/images_slices/'
    train_path_masks = base_path + 'train/Masks/'
    train_path_mpatches = base_path + 'train/masks_slices/'
    train_path_ann = base_path + 'train/Annotations/'

    val_path_images = base_path + 'val/Images/'
    val_path_ipatches = base_path + 'val/images_slices/'
    val_path_masks = base_path + 'val/Masks/'
    val_path_mpatches = base_path + 'val/masks_slices/'
    val_path_ann = base_path + 'val/Annotations/' 

    
    create_dir(train_path_images)
    create_dir(train_path_masks)
    create_dir(train_path_ipatches)
    create_dir(train_path_mpatches)
    create_dir(train_path_ann)

    create_dir(val_path_images)
    create_dir(val_path_masks)
    create_dir(val_path_ipatches)
    create_dir(val_path_mpatches)
    create_dir(val_path_ann)
    wrong_patients = [
                'RPE6506',
                'RPE6508',
                'RPE6509',
                'RPE6519',
                'RPE6520',
                'RPE6522',
                'RPE6523',
                'RPE6511'
    ]

    filenames_oct = get_filenames(OCT_PATH, 'vol')
    
    patients = []
    for fi in filenames_oct:
        if 'IRD_RPE65' in fi:
            patients.append(fi)

    train, val = train_test_split(filenames_oct, train_size=0.8, shuffle=True)

    for t in tqdm(train):
        name = os.path.splitext(os.path.split(t)[1])[0]
        if not str(name).split('_')[1] + str(name).split('_')[2] in wrong_patients or 'Anonym' in name:
            print(str(name))
            print(str(name).split('_')[1] + str(name).split('_')[2])
            oct_read = vol_files(name)
            data = oct_read.bscans[0].scan
            mask = do_dataset(t, data, name, train_path_images, train_path_masks, train_path_ann)
            slicing(image=data, mask=mask, name=name,
                    path_img=train_path_ipatches, path_msk=train_path_mpatches, size=128, shift=64)
    for v in tqdm(val):
        name = os.path.splitext(os.path.split(v)[1])[0]
        if not str(name).split('_')[1]  + str(name).split('_')[2] in wrong_patients  or 'Anonym' in name:
            print(str(name))
            print(str(name).split('_')[1] + str(name).split('_')[2])
            oct_read = vol_files(name)
            data = oct_read.bscans[0].scan
            mask = do_dataset(v, data, name, val_path_images, val_path_masks, val_path_ann)
            slicing(image=data, mask=mask, name=name,
                    path_img=val_path_ipatches, path_msk=val_path_mpatches, size=128, shift=64)

if __name__ == "__main__":
    main()
