import os
import numpy as np
import pandas as pd
import matplotlib
from PIL import Image
from tqdm import tqdm
from eyepy.core.base import Oct
from matplotlib import cm
from utils.utils import create_dir, get_filenames
from sklearn.model_selection import train_test_split
from utils.definitions import *

class OCTDataset:
    '''
    A class representing an OCT (Optical Coherence Tomography) dataset.

    Attributes:
        file (str): The path of the OCT file.
        oct_read (Oct): An instance of the Oct class representing the OCT data.
        patient_id (str): The ID of the patient associated with the OCT data.
        OPL (ndarray): The OPL (Outer Plexiform Layer) layer of the OCT data.
        INL (ndarray): The INL (Inner Nuclear Layer) layer of the OCT data.
        PR2 (ndarray): The PR2 (Photoreceptor Inner Segment) layer of the OCT data.
        PR1 (ndarray): The PR1 (Photoreceptor Outer Segment) layer of the OCT data.
        BM (ndarray): The BM (Bruch's Membrane) layer of the OCT data.
        ELM (ndarray): The ELM (External Limiting Membrane) layer of the OCT data.
        name (str): The name of the OCT dataset.
        bscan (ndarray): The B-scan image of the OCT data.
        data (ndarray): The data of the OCT scan.
        annotation (ndarray): The annotated image of the OCT data.
        mask (ndarray): The mask representing the ground truth of the OCT data.

    Methods:
        __init__(self, file=None): Initializes the OCTDataset object.
        open_vol_files(self): Opens the OCT volume files and extracts relevant information.
        get_annotations(self): Retrieves the annotations for different layers of the OCT data.
        get_dataset(self, path_img, path_msk, path_ann): Generates images, masks, and annotations from the OCT data.
        create_mask_rgb(self, mask): Creates an RGB mask from the given mask.
        blend_images(self, img_in, mask_rgb): Blends the input image and mask together.
        slicing(self, path_img_slice, path_msk_slice, size=128, shift=64): Slices the OCT data into smaller images.

    '''
        
    def __init__(self, file=None):
        self.file = file
        self.open_vol_files()
        self.get_annotations()
        
    def open_vol_files(self):
        '''
        Function to open the OCT volume files and extract relevant information.
        Args:
            None
        Return:
            None
        '''
        self.oct_read = Oct.from_heyex_vol(self.file)
        self.patient_id = str(self.oct_read.meta['PatientID'])
        self.bscan = self.oct_read.bscans[0].scan
        
    def get_annotations(self):
        '''
        Function to get the annotations for different layers of the OCT data.
        Args:
            None
        Return:
            None
        '''
        def get_layer(layer_name):
            default_shape = self.oct_read.bscans[0].scan.shape[1]
            try:
                layer = np.round(self.oct_read.bscans[0].annotation["layers"][layer_name]).astype('uint16')
            except:
                layer = np.zeros(default_shape, dtype='uint16')
                pass
            return layer

        self.OPL = get_layer('OPL')
        self.INL = get_layer('INL')
        self.PR2 = get_layer('PR2')
        self.PR1 = get_layer('PR1')
        self.BM = get_layer('BM')
        self.ELM = get_layer('ELM')

    def get_dataset(self, path_img, path_msk, path_ann):
        '''
        Function to get images and masks from the OCT files
        Args:
            file (str): The path of the OCT file.
            path_img (str): The directory path to save the image files.
            path_msk (str): The directory path to save the mask files.
            path_ann (str): The directory path to save the annotation files.
        Return:
            None
        '''
        self.name = os.path.splitext(os.path.split(self.file)[1])[0]
        
        self.data = np.expand_dims(self.bscan, axis=-1)

        zeros = np.zeros((self.data.shape[0], self.data.shape[1], 3)).astype('uint8')
        self.annotation = np.add(self.data, zeros)
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=4)
        # Generate Mask ground truth
        self.mask = np.zeros((self.data.shape[0], self.data.shape[1])).astype('float32')
        color1 = cm.hsv(norm(1), bytes=True)[:3]
        color2 = cm.hsv(norm(2), bytes=True)[:3]
        color3 = cm.hsv(norm(3), bytes=True)[:3]
        color4 = cm.hsv(norm(4), bytes=True)[:3]
        
        for i in range(self.BM.shape[0]):
            self.annotation[self.INL[i], i, :] = color2
            self.annotation[self.OPL[i], i, :] = color2
            self.annotation[self.ELM[i], i, :] = color3
            self.annotation[self.PR1[i], i, :] = color1
            self.annotation[self.PR2[i], i, :] = color1
            self.annotation[self.BM[i],  i, :] = color4
            # OPL
            self.mask[self.INL[i]:self.OPL[i], i] = 2 if self.INL[i] <= self.OPL[i] and self.INL[i] > 0 and self.OPL[i] > 0 else self.mask[self.INL[i]:self.OPL[i], i]
            # ELM
            self.mask[self.ELM[i]:self.PR1[i], i] = 3 if self.ELM[i] <= self.PR1[i] and self.ELM[i] > 0 and self.PR1[i] > 0 else self.mask[self.ELM[i]:self.PR1[i], i]
            # EZ
            self.mask[self.PR1[i]:self.PR2[i], i] = 1 if self.PR1[i] <= self.PR2[i] and self.PR1[i] > 0 and self.PR2[i] > 0 else self.mask[self.PR1[i]:self.PR2[i], i]
            # IZ-RPE
            self.mask[self.PR2[i]:self.BM[i], i] = 4 if self.PR2[i] <= self.BM[i] and self.PR2[i] > 0 and self.BM[i] > 0 else self.mask[self.PR2[i]:self.BM[i], i]
        mask_copy = np.copy(self.mask)
        self.mask_uint8 = np.multiply(mask_copy, 255).astype('uint8')
        mask_rgb = self.create_mask_rgb(self.mask)
        annotation = self.blend_images(self.annotation, mask_rgb)
        img_bscan = Image.fromarray(self.bscan)
        mask = Image.fromarray(self.mask_uint8)
        mask.convert('L')
        annotation = Image.fromarray(self.annotation)
        img_bscan.save(path_img + self.name + f".png")
        mask.save(path_msk + self.name + f".png")
        annotation.save(path_ann + self.name + f".png")

    def create_mask_rgb(self, mask):
        '''
        Function to create an RGB mask from the given mask.
        Args:
            mask (ndarray): The mask image.
        Return:
            mask_rgb (ndarray): The RGB mask image.
        '''
        shape_1 = (mask.shape[0], mask.shape[1], 3)
        mask_rgb = np.zeros(shape=shape_1, dtype='uint8')

        norm = matplotlib.colors.Normalize(vmin=0, vmax=mask.max())
        for idx in range(1, int(mask.max()) + 1):
            color = cm.hsv(norm(idx), bytes=True)
            mask_rgb[mask == idx] = color[:3]
        return mask_rgb
    
    def blend_images(self, img_in, mask_rgb):
        '''
        Function to blend the input image and mask together.
        Args:
            img_in (ndarray): The input image.
            mask_rgb (ndarray): The mask image.
        Return:
            overlay (Image): The blended image.
        '''
        img = Image.fromarray(img_in).convert("RGBA")
        img_blend = Image.fromarray(mask_rgb).convert("RGBA")
        overlay = Image.blend(img, img_blend, 0.5)
        return overlay
    
    def slicing(self, path_img_slice, path_msk_slice, size=128, shift=64):
        '''
        Function to slice the OCT data into smaller images.
        Args:
            path_img_slice (str): The directory path to save the image slices.
            path_msk_slice (str): The directory path to save the mask slices.
            size (int): The size of the slices.
            shift (int): The shift of the slices.
        return:
            None
        '''
        j = 1
        for i in range((size), (self.data.shape[1]), shift):
            img_save = self.bscan[:, i - size:i]
            msk_save = self.mask_uint8[:, i - size:i]
            img = Image.fromarray(img_save)
            msk = Image.fromarray(msk_save)
            img.save(path_img_slice + self.name + f"_{j}.png")
            msk.save(path_msk_slice + self.name + f"_{j}.png")
            j += 1

def oct_patient_train_split(filenames_oct):
    '''
    Function to split the dataset into training and validation sets.
    Args:
        filenames_oct (list): The list of OCT filenames.
    return:
        id_train (list): The list of training IDs.
        id_val (list): The list of validation IDs.
    '''
    IDs = []
    for idx, f1 in enumerate(tqdm(filenames_oct)):
        oct_read = Oct.from_heyex_vol(f1)
        patient_id = str(oct_read.meta['PatientID'])
        IDs.append(patient_id)
    
    id_unique = list(set(IDs))
    id_unique.sort()

    print(len(set(IDs)))
    
    s1, s2 = train_test_split(id_unique, shuffle=True, train_size=0.8, test_size=0.2)
    id_train = s1
    id_val = s2
    print('id_train = ', s1)
    print('id_train = ', s2)
    return id_train, id_val 

def main():
    base_path = BASE_PATH
    train_path_images = base_path + 'train/Images/'
    train_path_masks = base_path + 'train/Masks/'
    val_path_images = base_path + 'val/Images/'
    val_path_masks = base_path + 'val/Masks/'
    train_path_ann = base_path + 'train/Annotations/'
    val_path_ann = base_path + 'val/Annotations/'
    patches_images_train = base_path + 'train/images_slices/'
    patches_masks_train = base_path + 'train/masks_slices/'
    patches_images_val = base_path + 'val/images_slices/'
    patches_masks_val = base_path + 'val/masks_slices/'

    create_dir(train_path_images)
    create_dir(train_path_masks)
    create_dir(val_path_images)
    create_dir(val_path_masks)
    create_dir(patches_images_train)
    create_dir(patches_images_val)
    create_dir(patches_masks_train)
    create_dir(patches_masks_val)
    create_dir(train_path_ann)
    create_dir(val_path_ann)
    
    filenames_oct = get_filenames(OCT_DATA, 'vol')
    # filtered_filenames = [f for f in filenames_oct if 'IRD_RPE65' in f]
    id_train, id_val = oct_patient_train_split(filenames_oct)
    
    print(set(id_train) & set(id_val))
    df1 = pd.DataFrame({'Train IDs': id_train})
    df1.to_csv(base_path + 'train_patient_ids.csv')

    df1 = pd.DataFrame({'Val IDs': id_val})
    
    df1.to_csv(base_path + 'val_patient_ids.csv')

    print('Factor train_size: ', len(id_train) / (len(id_train) + len(id_val)))
    print(len(id_train), len(id_val))
    IDs = []
    N = 0
    sum_mean = 0.0
    squared_sum = 0.0
    

    for f1 in tqdm(filenames_oct):
        oct_data = OCTDataset(f1)
        sum_mean += oct_data.bscan.mean() / 255.
        squared_sum += (oct_data.bscan.mean() / 255. )**2
        
        N += 1
        IDs.append(oct_data.patient_id)
        if oct_data.patient_id in id_train:
            oct_data.get_dataset(train_path_images, train_path_masks, train_path_ann)
            oct_data.slicing(patches_images_train, patches_masks_train, size=128, shift=64)
        elif oct_data.patient_id in id_val:
            # print(oct_data.patient_id, ': ', 'Validation')
            oct_data.get_dataset(val_path_images, val_path_masks, val_path_ann)           
            oct_data.slicing(patches_images_val, patches_masks_val, size=128, shift=64)
        else:
            print('error')

    train_slices_count = len(os.listdir(patches_images_train))
    val_slices_count = len(os.listdir(patches_images_val))

    print("Number of files in train images_slices:", train_slices_count)
    print("Number of files in val images_slices:", val_slices_count)
    mean = sum_mean / N
    std = (squared_sum / N - mean ** 2) ** 0.5
    print(r'Mean:',mean, r'Std:',std)

if __name__ == "__main__":
    main()
