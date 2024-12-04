import os
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
from tqdm import tqdm
from utils.utils import get_filenames
from oct_library import OCTProcessing
import numpy as np

def main():
    base_path = 'logs/unet_03_13_17_58_31/'
    model_path = os.path.join(base_path, 'checkpoints/model.pth')
    model = torch.load(model_path, map_location='cuda')
    with open(os.path.join(base_path, 'experiment_cfg.yaml'), "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    # oct_path = 'dataset/1_bonn_dataset_test/VolumeFilesPatients/'
    # oct_path = 'dataset/1_bonn_dataset_test/Full_dataset/'
    # oct_path = 'dataset/1_bonn_dataset_test/VolumeFilesControls/'
    # oct_path = 'dataset/1_bonn_dataset_test/RPE65_VolumeFiles/'
    # oct_path = 'dataset/2_OCTAnnotated/'
    # oct_path = 'dataset/2_1_OCTLastVisit/'
    # oct_path = 'dataset/P011/'
    # oct_path = 'dataset/OCT_EZ/F_OCT'
    # oct_path =  'dataset/6_OCT_combined_annotatedLast/'
    oct_path = 'dataset/Analysis_dataset/'
    # oct_path = 'dataset/1_bonn_dataset_test/P019'
    oct_files = get_filenames(oct_path, 'vol')
    config_path = os.path.join(base_path, 'experiment_cfg.yaml')
    base_path1 = 'logs/BEST_LARGE/' #logs/BEST_0.94, logs/2023-01-22_02_39_26
    model_path1 = os.path.join(base_path1, 'checkpoints/model.pth')
    df = pd.DataFrame()
    diction = []
    temp_vol = ''
    for oct_file in tqdm(oct_files):
        vol_flag = True
        for etl in ['6mm', '3mm', '2mm','1mm', '0.5mm']: #, 
            try:
                oct_process = OCTProcessing(oct_file=oct_file, config_path=config_path, model_path=model_path) # 125, 36, 10, 68, 15
                oct_process.fovea_forward(ETDRS_loc=etl) #
                if vol_flag:
                    oct_process.volume_forward(model_path1, interpolated=True, tv_smooth=False, plot=False, bscan_positions=False)
                    temp_vol = oct_process.results['Volume_area']
                    temp_width = oct_process.results['EZ_diameter']
                    vol_flag = False
                diction.append(oct_process.results)
                oct_process.results['Volume_area'] = temp_vol
                oct_process.results['EZ_diameter'] = temp_width
            except Exception as exc:
                print("An exception occurred:", type(exc).__name__, "–", exc)
                continue
    df = pd.DataFrame(diction)
    df.to_csv(f'Luxurna_RPE65_v26_major.csv')

if __name__ == '__main__':
    main()