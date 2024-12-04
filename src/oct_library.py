import os
import yaml
import torch
import matplotlib 
import numpy as np
import albumentations as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datetime
import eyepy as ep

from scipy import ndimage
from PIL import Image
from matplotlib import cm
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from skimage.restoration import denoise_tv_chambolle
import pandas as pd

def predict(model, x_image, device):   
    
    MEAN =  0.1338 # 0 # 0.1338  # 0.1338 0.13505013393330723
    STD =  0.1466 # 1 # 0.1466  # 0.1466 0.21162075769722669
    normalization = T.Normalize(mean=MEAN, std=STD)

    n_dimention = np.ndim(x_image)
    # image = np.repeat(image, 3, axis=-1)
    image = normalization(image=x_image)
    image = np.expand_dims(image['image'], axis=-1)
    if n_dimention == 2:
        image = image.transpose((2, 0, 1))
    elif n_dimention == 3:
        image = image.transpose((0, 3, 1, 2))
    image = torch.tensor(image, dtype=torch.float, device=device)
    
    if torch.Tensor.dim(image) == 3:
        image = image.unsqueeze(0)

    y_pred = model(image)
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    if n_dimention == 2:
        y_pred = y_pred.squeeze(0)
    elif n_dimention == 3:
        y_pred = y_pred.squeeze(1)

    y_pred = y_pred.detach().cpu().numpy()
    return y_pred

class OCTProcessing:
    def __init__(self, oct_file, config_path, model_path, bscan_idx=None):
        
        with open(config_path, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
        self.imgh = cfg['general']['img_sizeh']
        self.imgw = cfg['general']['img_sizew']
        self.classes = cfg['general']['classes']
        self.mode = cfg['general']['img_type']
        self.oct_file = oct_file
        self.device = torch.device('cuda') # torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # torch.device('cpu') #   
        self.model = torch.load(model_path, map_location=self.device)
        self.gamma = cfg['preprocessing']['gamma']
        self.alphaTV = cfg['preprocessing']['alphatvd']
        self.per_batch = cfg['general']['per_batch']
        self.dataset = cfg['general']['dataset']
        
        
        # get metadata
        self.__read_oct(bscan_idx)  
        # get region of interest at 6mm
        self.__etdrs_loc()
        # get segmentation 
        self.pred_class_map, self.pred_rgb, self.overlay = self.__get_segmentation(self.model, self.bscan_fovea, self.mode, self.gamma, alpha=self.alphaTV)
        self.__get_roi(self.pred_class_map, offset=0)
        # get biomarkers in roi
        self.__get_layers_roi_intprof()
    
    def __load_surgery_dates(self, data_frame):
        # Extract relevant columns
        surgery_dates_df = data_frame[['P_ID', 'OD', 'OS']]
        
        # Convert the DataFrame to a dictionary
        surgery_dates_dict = surgery_dates_df.set_index('P_ID').T.to_dict()
        
        return surgery_dates_dict
        
    def __read_oct(self, bscan_idx=None):
        # read OCT
        self.oct = ep.import_heyex_vol(self.oct_file)
        self.foveax_pos = None
        if bscan_idx is not None:
            self.bscan_idx = bscan_idx
        else:
            self.bscan_idx = len(self.oct) // 2
        self.bscan_fovea = self.oct[self.bscan_idx].data
        try:
            self.pid_codes = pd.DataFrame(pd.read_excel('../dataset/20230325_Luxturna_surgerydates.xlsx'))
        except:
            self.pid_codes = pd.DataFrame(pd.read_excel('dataset/20230325_Luxturna_surgerydates.xlsx'))
                 
        self.surgery_dates = self.__load_surgery_dates(self.pid_codes)
        self.results = {}
        
        self.patient = os.path.splitext(os.path.split(self.oct_file)[1])[0].replace('-', '_').replace('RP65', 'IRD_RPE65')

        self.scale_x = self.oct.meta.as_dict()['scale_x']
        self.scale_y = self.oct.meta.as_dict()['scale_y'] # 0.010 if self.scale_x > 0.010 else 0.005
        self.pix_to_mm = 1 // self.scale_x

        self.pixel_2_mm2 = self.scale_x * self.scale_x
        self.visit_date = self.oct.meta.as_dict()['visit_date']
        self.laterality = self.oct.meta.as_dict()['laterality']
        self.dob = self.oct.meta.as_dict()['dob']
        self.loc_fovea = self.oct.meta.as_dict()['bscan_meta'][self.bscan_idx]['start_pos'][1] // self.scale_x
        self.fovea_xstart = self.oct.meta.as_dict()['bscan_meta'][self.bscan_idx]['start_pos'][0] // self.scale_x
        self.fovea_xstop = self.oct.meta.as_dict()['bscan_meta'][self.bscan_idx]['end_pos'][0] // self.scale_x
        date_format = '%Y-%m-%d'
        dos_date = datetime.datetime(1, 1, 1) 
        dos = dos_date.date()
        # dos = dos_date.strftime('%Y-%m-%d')
        if 'Anonym' in self.patient or 'Künzel' in self.patient:
            y = 'control'
        elif 'RP65' in self.patient or 'RPE65' in self.patient:
            print(self.patient)
            y = 'RPE65'
            pid_file = self.patient.split('_')
            filtered = self.pid_codes[self.pid_codes['RPE65_ID'].str.contains(pid_file[0] + '_' + pid_file[1] + '_' + pid_file[2])]
            
            if str(pid_file[0] + '_' + pid_file[1] + '_' + pid_file[2]) in self.pid_codes['RPE65_ID'].values:
                pid_new = filtered['P_ID'].values[0]  
            else:
                pid_new = self.patient
            print(pid_new)
            self.patient = self.patient.replace('IRD_RPE65_' + pid_file[2], str(pid_new))
            dos = self.surgery_dates[self.patient.split('_')[0]][str(self.laterality)]
            dos = dos.date()
        else:
            y = 'control'
        
        vd = str(self.visit_date).partition("T")[0]
        vd_clean = datetime.datetime.strptime(vd, date_format)
        today = datetime.datetime(2022, 12, 1)
        age = today.year - self.dob.year
        days_pt = int((vd_clean.date() - dos).days)

        months_pt = 1 if days_pt > 0 and days_pt < 42 else 3 \
            if days_pt >= 42 and days_pt < 131 else 6 \
            if days_pt >= 131 and days_pt < 221 else 9 \
            if days_pt >= 221 and days_pt < 310 else 12 \
            if days_pt >= 310 and days_pt < 403 else 15 \
            if days_pt >= 403 and days_pt < 480 else 18 \
            if days_pt >= 480 and days_pt < 582 else 24 \
            if days_pt >= 582 and days_pt < 2000 else 0 \
            if days_pt <= 0 and days_pt > -42 else -3 \
            if days_pt <= -42 and days_pt > -131 else np.nan \

        if self.foveax_pos is None:
            self.foveax_pos = (self.bscan_fovea.shape[1]) // 2

        self.results['PID'] = self.patient
        self.results["Patient"] = y
        self.results['Laterality'] = self.laterality
        self.results['DOB'] = self.dob.strftime('%Y-%m-%d')
        self.results['Age'] = 'pediatric' if age < 20 else 'adult'
        self.results['visit_date'] = vd
        self.results['DOS'] = dos
        self.results['MPT'] = months_pt
        self.results['N_Bscans'] = len(self.oct)
        self.results['Fovea_BScan'] = self.bscan_idx
        self.results['scale_y'] = self.scale_y
        self.results['scale_x'] = self.scale_x
        self.results['Y_Fovea'] = self.loc_fovea
        self.results['Fovea_xstart'] = self.fovea_xstart
        self.results['Fovea_xstop'] = self.fovea_xstop
        self.results['Volume_area'] = np.nan

    def __etdrs_loc(self):
        bscan_width = self.bscan_fovea.shape[1]
        self.outer_ring_max = min(int(self.foveax_pos + self.pix_to_mm * 3), bscan_width)
        self.outer_ring_min = max(int(self.foveax_pos - self.pix_to_mm * 3), 0)

        self.inner_ring_max = int(self.foveax_pos + self.pix_to_mm * 1.5)
        self.inner_ring_min = int(self.foveax_pos - self.pix_to_mm * 1.5)

        self.ring_2mm_max = int(self.foveax_pos + self.pix_to_mm)
        self.ring_2mm_min = int(self.foveax_pos - self.pix_to_mm)

        self.center_fovea_max = int(self.foveax_pos + self.pix_to_mm // 2)
        self.center_fovea_min = int(self.foveax_pos - self.pix_to_mm // 2)

        self.um5_max = int(self.foveax_pos + self.pix_to_mm // 4)
        self.um5_min = int(self.foveax_pos - self.pix_to_mm // 4)
    
    def __get_segmentation(self, model, img, mode, gamma=1, alpha=0.0001):
        img_in = tv_denoising(gray_gamma(img, gamma=gamma), alpha=alpha)
        pady = (np.abs(self.imgh - img_in.shape[0]) // 2) if self.imgh != img_in.shape[0] else 0
        if pady:
            img_in = np.pad(img_in, [(pady,), (0,)], 'constant', constant_values=0)

        if mode == 'large':
            preds = self.__predict_large_mode(model, img_in)
        elif mode == 'slices':
            preds = self.__predict_slices_mode(model, img_in)
            if pady:
                preds = preds[pady:img_in.shape[0]-pady, :]
                img_in = img_in[pady:img_in.shape[0]-pady, :]

        preds = closing_opening(preds)
        pred_rgb = self.__create_pred_rgb(preds)
        overlay = self.__blend_images(img_in, pred_rgb)

        self.bscan_fovea = img_in
        return preds, pred_rgb, overlay

    def __predict_large_mode(self, model, img_in):
        shape_image_x = img_in.shape
        image_x = F.interpolate(torch.from_numpy(img_in).unsqueeze(0).unsqueeze(0).float(), (self.imgh, self.imgw), mode='bilinear', align_corners=False).squeeze().numpy()
        pred = predict(model, image_x, self.device)
        preds = F.interpolate(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(), (shape_image_x[0], shape_image_x[1]), mode='nearest').squeeze().numpy()
        return preds

    def __predict_slices_mode(self, model, img_in):
        predictions = []
        if not self.per_batch:
            for i in range(self.imgw, img_in.shape[1] + self.imgw, self.imgw):
                image_x = img_in[:, i - self.imgw:i]
                predictions.append(predict(model, image_x, self.device).astype('uint8'))
        else:
            img_pred = [img_in[:, i - self.imgw:i] for i in range(self.imgw, img_in.shape[1] + self.imgw, self.imgw)]
            predictions = predict(model, np.array(img_pred), self.device)
        return np.hstack(predictions)

    def __create_pred_rgb(self, preds):
        shape_1 = (preds.shape[0], preds.shape[1], 3)
        pred_rgb = np.zeros(shape=shape_1, dtype='uint8')

        norm = matplotlib.colors.Normalize(vmin=0, vmax=preds.max())
        for idx in range(1, int(preds.max()) + 1):
            color = cm.hsv(norm(idx), bytes=True)
            pred_rgb[..., 0] = np.where(preds == idx, color[0], pred_rgb[..., 0])
            pred_rgb[..., 1] = np.where(preds == idx, color[1], pred_rgb[..., 1])
            pred_rgb[..., 2] = np.where(preds == idx, color[2], pred_rgb[..., 2])
        return pred_rgb

    def __blend_images(self, img_in, pred_rgb):
        img_overlay = Image.fromarray(img_in).convert("RGBA")
        pred_overlay = Image.fromarray(pred_rgb).convert("RGBA")
        overlay = Image.blend(img_overlay, pred_overlay, 0.4)
        return np.array(overlay)
    
    def __get_roi(self, mask, offset=2):
        mask_opl = np.where(mask == self.classes.index('OPL'), 1, 0)
        pos_opl = np.where(mask_opl)
        mask_ez = np.where(mask == self.classes.index('EZ'), 1, 0)
        pos_ez = np.where(mask_ez)
        mask_bm = np.where(mask == self.classes.index('BM'), 1, 0)
        pos_bm = np.where(mask_bm)
        try:
            ymin = np.min(pos_opl[0][np.nonzero(pos_opl[0])])
        except:
            ymin = np.min(pos_ez[0][np.nonzero(pos_opl[0])])

        try:
            ymax = np.max(pos_bm[0][np.nonzero(pos_bm[0])])
        except:
            ymin = np.max(pos_ez[0][np.nonzero(pos_opl[0])])
        

        if ymin < offset:
            ymin = 0
        else:
            ymin = ymin-offset

        if (ymax + offset) > mask.shape[0]:
            ymax = mask.shape[0]
        else:
            ymax = ymax + offset
        self.roi_pos = [ymin, ymax, int(self.outer_ring_min), int(self.outer_ring_max)]

    def __get_individual_layer(self, sample_bscan, sample_pred, layer, offset=0):
        binary_layer = get_layer_binary_mask(sample_pred, self.classes, layer=layer, offset=offset)
        segmented_layer = np.multiply(binary_layer, sample_bscan)
        return binary_layer, segmented_layer

    def __get_intensity_profiles(self, segmented_opl, segmented_ez, segmented_elm, binary_elm, binary_bm):
        # getting max peaks and ELM BM localization
        max_opl_x, max_opl_y = get_max_peak(segmented_opl)

        max_ez_x, max_ez_y = get_max_peak(segmented_ez)

        max_elm_x, max_elm_y = get_max_peak(segmented_elm)

        lim_elm_y = get_limit(binary_elm, side='min', offset=2)

        lim_bm_y = get_limit(binary_bm, side='max')

        return max_opl_x, max_opl_y, max_ez_x, max_ez_y, lim_elm_y, lim_bm_y
    
    def __get_layers_roi_intprof(self):
        roi_slices = (slice(self.roi_pos[0], self.roi_pos[1]), slice(self.roi_pos[2], self.roi_pos[3]))
        self.bscan_roi = self.bscan_fovea[roi_slices]
        self.pred_roi = self.pred_class_map[roi_slices]
        self.pred_rgb_roi = self.pred_rgb[roi_slices]
        self.overlay_roi = self.overlay[roi_slices]

        layers = ['OPL', 'ELM', 'EZ', 'BM']
        offsets = [0, 0, 1, 0]
        binary_layers = []
        segmented_layers = []

        for layer, offset in zip(layers, offsets):
            binary, segmented = self.__get_individual_layer(self.bscan_roi, self.pred_roi, layer, offset)
            binary_layers.append(binary)
            segmented_layers.append(segmented)

        self.opl_binary_roi, self.elm_binary_roi, self.ez_binary_roi, self.bm_binary_roi = binary_layers
        self.opl_roi, self.elm_roi, self.ez_roi, self.bm_roi = segmented_layers

        self.binary_total = np.where((self.pred_roi.astype(float) <= 4) & (self.pred_roi.astype(float) > 0), 1, 0)
        self.segmented_total = np.multiply(self.binary_total, self.bscan_roi)

        self.max_opl_x, self.max_opl_y, self.max_ez_x, self.max_ez_y, self.lim_elm, self.lim_bm = self.__get_intensity_profiles(
            self.opl_roi, self.ez_roi, self.elm_roi, self.elm_binary_roi, self.bm_binary_roi)
    
    def __get_biomarkers(self, segmented_ez, binary_ez,  max_opl_y, max_ez_y, lim_elm_y, lim_bm_y):
        # *********** compute bio-markers *********** 
        rezi_mean, rezi_std = get_rEZI(max_ez_y, max_opl_y) # RELATIVE EZ INSTENSITY
        ez_th_mean, ez_th_std = get_thickness(binary_ez, self.scale_y) # EZ THICKNESS
        opl_2_ez_mean, opl_2_ez_std = get_distance_in_mm(max_opl_y, max_ez_y, self.scale_y) # DISTANCE FROM OPL TO EZ PEAKS IN MM
        elm_2_ez_mean, elm_2_ez_std = get_distance_in_mm(lim_elm_y, max_ez_y, self.scale_y) # DISTANCE FROM ELM TO EZ PEAKS IN MM
        ez_2_bm_mean, ez_2_bm_std = get_distance_in_mm(max_ez_y, lim_bm_y, self.scale_y) # DISTANCE FROM EZ TO BM PEAKS IN MM
        elm_2_bm_mean, elm_2_bm_std = get_distance_in_mm(lim_elm_y, lim_bm_y, self.scale_y) # DISTANCE FROM ELM TO BM PEAKS IN MM
        _, ez_tv_mean, ez_tv_std = get_total_variation(segmented_ez, 3) # TOTAL VARIATION
        return [rezi_mean, ez_th_mean, opl_2_ez_mean, elm_2_ez_mean, ez_2_bm_mean, elm_2_bm_mean, ez_tv_mean], \
               [rezi_std, ez_th_std, opl_2_ez_std, elm_2_ez_std, ez_2_bm_std, elm_2_bm_std, ez_tv_std]
    
    def fovea_forward(self, ETDRS_loc='6mm'):
        
        if ETDRS_loc == '0.5mm':
            self.__lateral_ref = [int(self.bscan_roi.shape[1] // 2 - self.pix_to_mm // 4), int(self.bscan_roi.shape[1] // 2 + self.pix_to_mm // 4)]
        elif ETDRS_loc == '1mm':
            self.__lateral_ref = [int(self.bscan_roi.shape[1] // 2 - self.pix_to_mm // 2), int(self.bscan_roi.shape[1] // 2 + self.pix_to_mm // 2)]
        elif ETDRS_loc == '2mm':
            self.__lateral_ref = [int(self.bscan_roi.shape[1] // 2 - self.pix_to_mm // 1), int(self.bscan_roi.shape[1] // 2 + self.pix_to_mm // 1)]
        elif ETDRS_loc == '3mm':
            self.__lateral_ref = [int(self.bscan_roi.shape[1] // 2 - self.pix_to_mm * 1.5), int(self.bscan_roi.shape[1] // 2 + self.pix_to_mm * 1.5)]
        elif ETDRS_loc == '6mm':
            self.__lateral_ref = [0, self.bscan_roi.shape[1]]
        else:
            raise Exception("ETDRs diameter not implemented, expected: [0.5mm, 1mm, 2mm, 3mm, 6mm]!")
        # if ETDRS_loc == '0.5mm':
        #     self.__lateral_ref = [self.um5_min, self.um5_max]
        # elif ETDRS_loc == '1mm':
        #     self.__lateral_ref = [self.center_fovea_min, self.center_fovea_max]
        # elif ETDRS_loc == '2mm':
        #     self.__lateral_ref = [self.ring_2mm_min, self.ring_2mm_max]
        # elif ETDRS_loc == '3mm':
        #     self.__lateral_ref = [self.inner_ring_min, self.inner_ring_max]
        # elif ETDRS_loc == '6mm':
        #     self.__lateral_ref = [0, self.bscan_roi.shape[1]]
        # else:
        #     raise Exception("ETDRs diameter not implemented, expected: [0.5mm, 1mm, 2mm, 3mm, 6mm]!")

        for ref in ['', '_nasal', '_temporal']:
            if (ref == '_nasal' and self.laterality == 'OS') or (ref == '_temporal' and self.laterality == 'OD'):
                max_opl_y = self.max_opl_y[self.__lateral_ref[0]:self.bscan_roi.shape[1] // 2]
                max_ez_y = self.max_ez_y[self.__lateral_ref[0]:self.bscan_roi.shape[1] // 2]
                lim_elm_y = self.lim_elm[self.__lateral_ref[0]:self.bscan_roi.shape[1] // 2]
                lim_bm_y = self.lim_bm[self.__lateral_ref[0]:self.bscan_roi.shape[1] // 2]
                segmented_ez = self.ez_roi[:, self.__lateral_ref[0]:self.bscan_roi.shape[1] // 2]
                binary_ez = self.ez_binary_roi[:, self.__lateral_ref[0]:self.ez_binary_roi.shape[1] // 2]
            elif (ref == '_nasal' and self.laterality == 'OD') or (ref == '_temporal' and self.laterality == 'OS'):
                max_opl_y = self.max_opl_y[self.bscan_roi.shape[1] // 2:self.__lateral_ref[1]]
                max_ez_y = self.max_ez_y[self.bscan_roi.shape[1] // 2:self.__lateral_ref[1]]
                lim_elm_y = self.lim_elm[self.bscan_roi.shape[1] // 2:self.__lateral_ref[1]]
                lim_bm_y = self.lim_bm[self.bscan_roi.shape[1] // 2:self.__lateral_ref[1]]
                segmented_ez = self.ez_roi[:, self.bscan_roi.shape[1] // 2:self.__lateral_ref[1]]
                binary_ez = self.ez_binary_roi[:, self.ez_binary_roi.shape[1] // 2:self.__lateral_ref[1]]
            else:
                # raise Exception("Lateral ref allowed only 'nasal' or 'temporal'!")
                max_opl_y = self.max_opl_y[self.__lateral_ref[0]:self.__lateral_ref[1]]
                max_ez_y = self.max_ez_y[self.__lateral_ref[0]:self.__lateral_ref[1]]
                lim_elm_y = self.lim_elm[self.__lateral_ref[0]:self.__lateral_ref[1]]
                lim_bm_y = self.lim_bm[self.__lateral_ref[0]:self.__lateral_ref[1]]
                segmented_ez = self.ez_roi[:, self.__lateral_ref[0]:self.__lateral_ref[1]]
                binary_ez = self.ez_binary_roi[:, self.__lateral_ref[0]:self.__lateral_ref[1]]
            biomark_mean, biomark_std = self.__get_biomarkers(segmented_ez, binary_ez, max_opl_y, max_ez_y, lim_elm_y, lim_bm_y)
            
            biomarkers = {
                "ETDRS_loc":ETDRS_loc,
                f'rEZI_mean{ref}':biomark_mean[0],
                f'EZ_th_mean{ref}':biomark_mean[1],
                f'EZ_OPL_mean{ref}':biomark_mean[2],
                f'EZ_ELM_mean{ref}':biomark_mean[3],
                f'EZ_BM_mean{ref}':biomark_mean[4],
                f'ELM_BM_mean{ref}':biomark_mean[5],
                f'EZ_TV_mean{ref}':biomark_mean[6],
                f'rEZI_std{ref}':biomark_std[0],
                f'EZ_th_std{ref}':biomark_std[1],
                f'EZ_OPL_std{ref}':biomark_std[2],
                f'EZ_ELM_std{ref}':biomark_std[3],
                f'EZ_BM_std{ref}':biomark_std[4],
                f'ELM_BM_std{ref}':biomark_std[5],
                f'EZ_TV_std{ref}':biomark_std[6],
            }
            self.results.update(biomarkers)
            # print(self.patient, f'\t', self.laterality, f'\t\t', ETDRS_loc, f'\t\t{np.round(biomark_mean[0],2)}\t\t', f'{np.round(biomark_mean[1],2)}\t\t', f'{np.round(biomark_mean[2],2)}\t\t',
            #                                      f'{np.round(biomark_mean[3],2)}\t\t', f'{np.round(biomark_mean[4],2)}\t\t', f'{np.round(biomark_mean[5],2)}\t\t', f'{np.round(biomark_mean[6],2)}')
    
    def volume_forward(self, big_model_path=None, interpolated=True, tv_smooth=False, plot=False, bscan_positions=True):
        X_MINS = []
        X_MAX = []
        Y_POS = []

        delta_ez_lim = []

        if big_model_path is not None:
            model = torch.load(big_model_path, map_location='cuda')
            mode = 'large'
        else:
            model = self.model
            mode = self.mode
        bscan1 = self.oct[len(self.oct) // 2].data
        pred_class_map1, _, _ = self.__get_segmentation(model, bscan1, mode, gamma=self.gamma, alpha=self.alphaTV)
        pred_class_map1 = pred_class_map1[self.roi_pos[0]:self.roi_pos[1],self.roi_pos[2]:self.roi_pos[3]]
        ez_mask1 = get_layer_binary_mask(pred_class_map1, self.classes, layer='EZ', offset=2)
        pos_ez1 = np.where(ez_mask1)
        try:
            ez_xmin_fovea = np.min(pos_ez1[1][np.nonzero(pos_ez1[1])])
            ez_xmax_fovea = np.max(pos_ez1[1][np.nonzero(pos_ez1[1])])
        except:
            ez_xmin_fovea = 0
            ez_xmax_fovea = 0
        ez_fovea_width = np.abs(ez_xmin_fovea - ez_xmax_fovea) * self.scale_x
        self.results['EZ_diameter'] = ez_fovea_width
        if len(self.oct) > 1:
            for idx in range(len(self.oct)):
                bscan = self.oct[idx].data
                xstart = self.oct.meta.as_dict()['bscan_meta'][idx]['start_pos'][0]//self.oct.meta.as_dict()['scale_x']
                y_position = self.oct.meta.as_dict()['bscan_meta'][idx]['start_pos'][1] // self.oct.meta.as_dict()['scale_x']
                # try:
                pred_class_map, _, _ = self.__get_segmentation(model, bscan, mode, gamma=self.gamma, alpha=self.alphaTV)
                pred_class_map = pred_class_map[self.roi_pos[0]:self.roi_pos[1],self.roi_pos[2]:self.roi_pos[3]]
                
                ez_mask = get_layer_binary_mask(pred_class_map, self.classes, layer='EZ', offset=0)
                pos_ez = np.where(ez_mask)
                try:
                    xmin = self.roi_pos[2] + np.min(pos_ez[1][np.nonzero(pos_ez[1])])   
                except:
                    xmin = 0
                try:
                    xmax = self.roi_pos[2] + np.max(pos_ez[1][np.nonzero(pos_ez[1])])
                except:
                    xmax = 0
                if (xmax - xmin) > 20:
                    # xmin = 0
                    # xmax = 0
                    delta_ez_lim.append(np.abs(xmax * self.scale_x - xmin * self.scale_x))
                    X_MINS.append(xmin + xstart)
                    X_MAX.append(xmax + xstart)
                    Y_POS.append(y_position)
                # except Exception as exc:
                #     print("An exception occurred:", type(exc).__name__, "–", exc)
                #     continue
            # print(int(Y_POS[0] - Y_POS[-1]))
            try:
                if interpolated and len(self.oct) > 1:
                    ynew1 = np.linspace(Y_POS[0], Y_POS[-1], num=int(Y_POS[0] - Y_POS[-1]))
                    f1 = interp1d(Y_POS, X_MAX, kind='linear', fill_value="extrapolate")
                    ynew2 = np.linspace(Y_POS[0], Y_POS[-1], num=int(Y_POS[0] - Y_POS[-1]))
                    f2 = interp1d(Y_POS, X_MINS, kind='linear', fill_value="extrapolate")
                    func1 = f1(ynew1)
                    func2 = f2(ynew2)
                    array_diff = np.array(func1) - np.array(func2)
                    volume_area = np.sum(array_diff) * self.pixel_2_mm2
            except:
                volume_area = np.nan
        else:
            Y_POS = self.oct.meta.as_dict()['bscan_meta'][0]['start_pos'][1] // self.oct.meta.as_dict()['scale_x']
            X_MINS = ez_xmin_fovea
            X_MAX = ez_xmax_fovea
            volume_area = np.nan
        # print(volume_area)
            
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=1, dpi=120, figsize=(8,8), frameon=False)
            self.oct.plot(localizer=True, bscan_positions=bscan_positions, bscan_region=True)
            if tv_smooth and len(self.oct) > 1 and interpolated:
                # assert interpolated, 'Interpolation for Total variation 1-D smooth is needed -> arg* interpolated = True'
                YNEW1 = denoising_1D_TV(f1(ynew1), 2)
                ax.plot(YNEW1, ynew1, '-', c='y', linewidth=2)
                YNEW2 = denoising_1D_TV(f2(ynew2), 2)
                ax.plot(YNEW2, ynew2, '-', c='y', linewidth=2)
                # print(YNEW1[0], YNEW2[0])
                ax.plot([YNEW1[0], YNEW2[0]], [ynew2[0], ynew1[0]],  '-', c='y', linewidth=2)
                ax.plot([YNEW1[-1], YNEW2[-1]], [ynew2[-1], ynew1[-1]], '-', c='y', linewidth=2)
                ax.plot([self.fovea_xstart, self.fovea_xstop], [self.foveax_pos, self.foveax_pos], c='lime', linewidth=3)
                # plt.legend(loc='best')
            # self.plot_etdrs_grid(ax)
            ax.tick_params(labelsize=12)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            # ax.legend(loc='best')
            fig.tight_layout()
        self.results['Volume_area'] = volume_area

    def plot_etdrs_grid(self, ax):
        pix_to_mm = 1 // self.scale_x
        r1 = int(pix_to_mm * 3)  # 3 mm ratio
        r2 = r1/2
        r3 = r2/2
        r4 = r3/2
        r5 = int(pix_to_mm * 1) 

        foveax_pos = (self.oct.shape[1]) // 2
        if self.oct.localizer.shape[0] == self.bscan_fovea.shape[1]:
            center = int(foveax_pos*1.55) 
        else:
            center = int(foveax_pos*3.141)
        theta = np.linspace(0, 2*np.pi, 100)
        d45 = np.pi / 4
        
        # radio * constant + x_point
        
        x1 = center + r1 * np.cos(theta)
        y1 = center + r1 * np.sin(theta)

        
        x2 = center + r2 * np.cos(theta)
        y2 = center + r2 * np.sin(theta)

        
        x3 = center + r3 * np.cos(theta)
        y3 = center + r3 * np.sin(theta)
        
        
        x4 = center + r4 * np.cos(theta)
        y4 = center + r4 * np.sin(theta)

        
        x5 = center + r5 * np.cos(theta)
        y5 = center + r5 * np.sin(theta)

        # Quad I
        x_pt1 = center + r1 * np.cos(d45)
        y_pt1 = center + r1 * -np.sin(d45)
        x_pt2 = center + r3 * np.cos(d45)
        y_pt2 = center + r3 * -np.sin(d45)

        # Quad II
        x_pt3 = center + r1 * -np.cos(d45)
        y_pt3 = center + r1 * -np.sin(d45)
        x_pt4 = center + r3 * -np.cos(d45)
        y_pt4 = center + r3 * -np.sin(d45)

        # Quad III
        x_pt5 = center + r1 * -np.cos(d45)
        y_pt5 = center + r1 * np.sin(d45)
        x_pt6 = center + r3 * -np.cos(d45)
        y_pt6 = center + r3 * np.sin(d45)

        # Quad IV
        x_pt7 = center + r1 * np.cos(d45)
        y_pt7 = center + r1 * np.sin(d45)
        x_pt8 = center + r3 * np.cos(d45)
        y_pt8 = center + r3 * np.sin(d45)
        ax.plot([x_pt1, x_pt2], [y_pt1, y_pt2], c='lime')
        ax.plot([x_pt3, x_pt4], [y_pt3, y_pt4], c='lime')
        ax.plot([x_pt5, x_pt6], [y_pt5, y_pt6], c='lime')
        ax.plot([x_pt7, x_pt8], [y_pt7, y_pt8], c='lime')
        ax.plot(x1,y1, c='lime')
        ax.plot(x2,y2, c='lime')
        # ax.plot(x5,y5, '--',c='lime')
        ax.plot(x3,y3, c='lime')
        # ax.plot(x4,y4, '--',c='lime')

    def plot_overlay_oct_segmentation(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(14,10), gridspec_kw={'width_ratios': [1]}, frameon=False)
        ax.set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax.set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax.tick_params(labelsize=12)
        ax.tick_params(labelsize=12)
        ax.imshow(self.overlay)

    def plot_slo_etdrs(self):
        pix_to_mm = 1 // self.scale_x
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(25,10), frameon=False) #gridspec_kw={'width_ratios': [1, 2]}
        self.oct.plot(localizer=True, bscan_positions=False, ax=ax)
        self.plot_etdrs_grid(ax)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.tick_params(labelsize=12)

    def plot_slo_fovea(self, etdrs_grid=False):
        pix_to_mm = 1 // self.scale_x

        fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(25,10), gridspec_kw={'width_ratios': [1, 1.8]}, frameon=False) #gridspec_kw={'width_ratios': [1, 2]}
        self.oct.plot(localizer=True, bscan_positions=True, ax=ax[0], bscan_region=True)
        ax[1].imshow(self.bscan_fovea, cmap='gray')
        ax[0].scatter(20+self.fovea_xstart, self.loc_fovea, c='red', s=50)
        ax[0].scatter(self.bscan_fovea.shape[1]-20+self.fovea_xstart, self.loc_fovea, c='red', s=50)
        if etdrs_grid:
            self.plot_etdrs_grid(ax[0])
        ax[0].legend(loc='best')
        ax[1].set_xlabel('B-Scan (X)', fontsize=24, weight="bold")
        ax[1].set_ylabel('A-Scan (Y)', fontsize=24, weight="bold")
        ax[0].set_xlabel('B-Scan (X)', fontsize=24, weight="bold")
        ax[0].set_ylabel('Volume (Z)', fontsize=24, weight="bold")
        ax[0].tick_params(labelsize=20)
        ax[0].tick_params(labelsize=20)
        ax[1].tick_params(labelsize=20)
        ax[1].tick_params(labelsize=20)

    def plot_segmentation_localization(self, etdrs=False):
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,8), dpi=200, frameon=False)
        ax.imshow(self.overlay, cmap='gray')
        
        if etdrs:
            ax.plot([self.um5_min, self.um5_min], [self.roi_pos[0], self.roi_pos[1]], '--', linewidth=1, color='lime', label='0.5mm')
            ax.plot([self.um5_max, self.um5_max], [self.roi_pos[0], self.roi_pos[1]], '--', linewidth=1, color='lime')
            ax.plot([self.um5_min, self.um5_max], [self.roi_pos[1], self.roi_pos[1]], '--', linewidth=1, color='lime')
            ax.plot([self.um5_min, self.um5_max], [self.roi_pos[0], self.roi_pos[0]], '--', linewidth=1, color='lime')

            ax.plot([self.center_fovea_min, self.center_fovea_min], [self.roi_pos[0], self.roi_pos[1]], linewidth=1, color='lime', label='1mm')
            ax.plot([self.center_fovea_max, self.center_fovea_max], [self.roi_pos[0], self.roi_pos[1]], linewidth=1, color='lime')
            ax.plot([self.center_fovea_min, self.center_fovea_max], [self.roi_pos[1], self.roi_pos[1]], linewidth=1, color='lime')
            ax.plot([self.center_fovea_min, self.center_fovea_max], [self.roi_pos[0], self.roi_pos[0]], linewidth=1, color='lime')

            ax.plot([self.ring_2mm_min, self.ring_2mm_min], [self.roi_pos[0], self.roi_pos[1]], '--', linewidth=1, color='lime', label='2mm')
            ax.plot([self.ring_2mm_max, self.ring_2mm_max], [self.roi_pos[0], self.roi_pos[1]], '--', linewidth=1, color='lime')
            ax.plot([self.ring_2mm_min, self.ring_2mm_max], [self.roi_pos[1], self.roi_pos[1]], '--', linewidth=1, color='lime')
            ax.plot([self.ring_2mm_min, self.ring_2mm_max], [self.roi_pos[0], self.roi_pos[0]], '--', linewidth=1, color='lime')

            ax.plot([self.inner_ring_min, self.inner_ring_min], [self.roi_pos[0], self.roi_pos[1]], linewidth=1, color='lime', label='3mm')
            ax.plot([self.inner_ring_max, self.inner_ring_max], [self.roi_pos[0], self.roi_pos[1]], linewidth=1, color='lime')
            ax.plot([self.inner_ring_min, self.inner_ring_max], [self.roi_pos[1], self.roi_pos[1]], linewidth=1, color='lime')
            ax.plot([self.inner_ring_min, self.inner_ring_max], [self.roi_pos[0], self.roi_pos[0]], linewidth=1, color='lime')

            ax.plot([self.outer_ring_min, self.outer_ring_min], [self.roi_pos[0], self.roi_pos[1]], linewidth=1, color='lime', label='6mm')
            ax.plot([self.outer_ring_max, self.outer_ring_max], [self.roi_pos[0], self.roi_pos[1]], linewidth=1, color='lime')
            ax.plot([self.outer_ring_min, self.outer_ring_max], [self.roi_pos[1], self.roi_pos[1]], linewidth=1, color='lime')
            ax.plot([self.outer_ring_min, self.outer_ring_max], [self.roi_pos[0], self.roi_pos[0]], linewidth=1, color='lime')
        # ax.set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        # ax.set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.tick_params(labelsize=12)
        # ax.legend(loc='best')
        figure.tight_layout()

    def plot_results_roi(self):
        bscan_roi = self.bscan_roi[:, self.__lateral_ref[0]:self.__lateral_ref[1]]
        overlay_roi = self.overlay_roi[:, self.__lateral_ref[0]:self.__lateral_ref[1]]
        max_opl_y = self.max_opl_y[self.__lateral_ref[0]:self.__lateral_ref[1]]
        max_ez_y = self.max_ez_y[self.__lateral_ref[0]:self.__lateral_ref[1]]
        lim_elm = self.lim_elm[self.__lateral_ref[0]:self.__lateral_ref[1]]
        lim_bm = self.lim_bm[self.__lateral_ref[0]:self.__lateral_ref[1]]
        max_opl_y = max_opl_y.copy().astype('float')
        max_opl_y[max_opl_y == 0] = np.nan
        max_ez_y = max_ez_y.copy().astype('float')
        max_ez_y[max_ez_y == 0] = np.nan
        lim_elm = lim_elm.copy().astype('float')
        lim_elm[lim_elm == 0] = np.nan
        lim_bm = lim_bm.copy().astype('float')
        lim_bm[lim_bm == 0] = np.nan
        figure, ax = plt.subplots(nrows=2, ncols=1, figsize=(14,8), frameon=True, dpi=200)
        ax[0].imshow(overlay_roi, cmap='gray')
        ax[1].imshow(bscan_roi, cmap='gray')
        ax[1].plot(max_opl_y, c='cyan', label='Max Peaks OPL')
        ax[1].plot(max_ez_y, linewidth=1.5,c='lime', label='Max Peaks EZ')
        # ax[1].plot(self.max_opl_x,self.y_opl_den, linewidth=1.5,c='cyan', label='Max Peaks EZ')
        # ax[1].plot(self.max_ez_x, self.y_ez_den, linewidth=1.5,c='lime', label='Max Peaks EZ')
        # ax[2].plot(self.max_elm_x, self.max_elm_y, c='violet', label='ELM')
        ax[1].plot(lim_elm, c='violet', label='ELM')
        ax[1].plot(lim_bm, c='red', label='BM')
        # ax[2].legend(loc='best')
        # ax[0].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        # ax[1].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        # ax[2].set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        # ax[2].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].axis('off')
        ax[1].axis('off')
        # ax[0].tick_params(labelsize=12)
        # ax[1].tick_params(labelsize=12)
        # ax[2].tick_params(labelsize=12)
        # ax[2].tick_params(labelsize=12)
        figure.tight_layout()

    def plot_total_variation_alphas(self, ax, alphas=[0.005, 0.05, 0.5], beta=3, xlabel=False):
        p = np.percentile(self.bscan_roi[:, self.__lateral_ref[0]:self.__lateral_ref[1]], 95)
        sample_bscan1 = self.bscan_roi[:, self.__lateral_ref[0]:self.__lateral_ref[1]] / p

        EZ_segmented1 = np.multiply(self.ez_binary_roi[:, self.__lateral_ref[0]:self.__lateral_ref[1]+35], sample_bscan1)
        locales, mean_tv, _ = get_total_variation(EZ_segmented1, beta)
        # plt.rc('font', size=30)
        locales[locales==0] = np.nan
        # ax.scatter(np.arange(locales.shape[0]), locales, s=5)
        ax.plot(locales*10000,  label=r'Original: ', linewidth=1, c='k')
        locales = 0
        clrs = ['r', 'g', 'b']
        ci = 0
        for w in alphas:
            tv_denoised = denoise_tv_chambolle(sample_bscan1, weight=w)
            EZ_segmented1 = np.multiply(self.ez_binary_roi[:, self.__lateral_ref[0]:self.__lateral_ref[1]], tv_denoised)
            locales, mean_tv, _ = get_total_variation(EZ_segmented1, beta)
            locales[locales==0] = np.nan
            ax.plot(locales*10000, '--', linewidth=0.8, label= r'$\alpha$: ' +  format(w,'.3f'), c=clrs[ci]) #/*
            ci+=1
        # if xlabel:
        #     ax.set_xlabel(r'N/(2$\beta$+1)', fontsize=32, weight="bold")
        # x.set_xlabel(r'$\beta$)', fontsize=16)
        # ax.set_ylabel('LV', fontsize=36, weight="bold")
        
        ax.locator_params(axis='y', nbins=6)
        ax.locator_params(axis='x', nbins=6)
        ax.grid(True)
        ax.set_xlim([0, locales.shape[0]])
        
        
        # ax.legend(bbox_to_anchor=(1, 0.85), loc="lower right", fontsize="30", ncol=2) #
        # plt.legend(loc="right", fontsize="32", ncol=1)
        return ax
        
    def plot_intensity_profiles(self, shift=1000, interpolation_3d=False):
        # p = np.percentile(self.bscan_fovea[self.references[2]:self.references[3], self.references[0]:self.references[1]], 95)
        sample_bscan1 = self.bscan_roi / 255.
        segmented_total1 = np.multiply(self.binary_total, sample_bscan1)
        img = segmented_total1
        int_prof_x = []
        size = 1

        for i in range(size, img.shape[1], size):
            window = img[:, i - size:i]
            matrix_mean = np.zeros((img.shape[0]))
            for j in range(window.shape[0]):
                matrix_mean[j] = window[j, :].mean()
            int_prof_x.append(matrix_mean)

        # for t in range(240, np.array(int_prof_x).shape[0], shift):
        # print(np.array(int_prof_x).shape)
        intensity = np.array(int_prof_x)[240, :]
        peaks, _ = find_peaks(intensity, height=0)
        y = np.arange(img.shape[0])
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figheight(4)
        fig.set_figwidth(10)
        ax.plot(intensity, y, 'k')
        ax.plot(intensity[peaks], peaks, "o", c='green')
        ax.set_xlabel('Grey value', fontsize=14, weight="bold")
        ax.set_ylabel(f'A-Scan \nDistance [Pixels]', fontsize=14, weight="bold")
        plt.gca().invert_yaxis()
        plt.show()
        if interpolation_3d:
            Z_INT = np.array(int_prof_x)
            x_int = np.arange(Z_INT.shape[1])
            y_int = np.arange(Z_INT.shape[0])
            X_INT, Y_INT = np.meshgrid(x_int, y_int)
            fig = plt.figure(figsize=(14, 8), dpi=200, frameon=False)
            ax = plt.axes(projection='3d')
            ax.set_aspect(aspect='auto', adjustable='datalim')
            ax.contour3D(X_INT, Y_INT, Z_INT, 100, cmap='jet')
            ax.set_xlabel('A-Scan(Y)', fontsize=14, weight="bold")
            ax.set_ylabel('B-Scan(X)', fontsize=14, weight="bold")
            ax.set_zlabel('Grey value', fontsize=14, weight="bold")
            ax.view_init(35, -95)    

def closing_opening(pred):
    iter = 1
    one_hot = np.eye(5)[pred]
    
    # one_hot[:, :, 3] = np.array(ndimage.binary_opening(one_hot[:, :, 3], iterations=1), dtype=type(pred))
    one_hot[:, :, 3] = np.array(ndimage.binary_closing(one_hot[:, :, 3], iterations=1), dtype=type(pred))

    # one_hot[:, :, 4] = np.array(ndimage.binary_opening(one_hot[:, :, 4], iterations=1), dtype=type(pred))
    one_hot[:, :, 4] = np.array(ndimage.binary_closing(one_hot[:, :, 4], iterations=1), dtype=type(pred))

    # one_hot[:, :, 2] = np.array(ndimage.binary_opening(one_hot[:, :, 2], iterations=1), dtype=type(pred))
    one_hot[:, :, 2] = np.array(ndimage.binary_closing(one_hot[:, :, 2], iterations=1), dtype=type(pred))

    # one_hot[:, :, 1] = np.array(ndimage.binary_opening(one_hot[:, :, 1], iterations=1), dtype=type(pred))
    one_hot[:, :, 1] = np.array(ndimage.binary_closing(one_hot[:, :, 1], iterations=1), dtype=type(pred))

    one_hot[:, :, 0] = np.array(ndimage.binary_closing(one_hot[:, :, 0], iterations=1), dtype=type(pred))
    
    amax_preds = np.argmax(one_hot, axis=2)

    return amax_preds

def get_max_peak(img: np.array, window_size=1):
    max1 = []
    size = window_size
    k = 0
    for i in range(size, img.shape[1] + 1, size):
        indices = (-img[:, i - size:i].reshape(-1)).argsort()[:1]
        row1 = (int)(indices[0] / size)
        col1 = indices[0] - (row1 * size)
        temp1 = row1, col1 + k
        k += size
        max1.append(temp1)
    max1 = np.array(max1)
    x1 = max1[:, 1]
    y1 = max1[:, 0]
    return np.array(x1), np.array(y1)

def get_limit(binary_mask, side, offset=0):
    size = 1
    lim_y = []
    
    for i in range(size, binary_mask.shape[1] + 1, size):
        col = binary_mask[:, i - size:i]
        if 1 in col:
            if side == 'max':
                lim_y.append(np.max(np.where(col)[0]) + offset)
            if side == 'min':
                lim_y.append(np.min(np.where(col)[0]) + offset)
        else:
            lim_y.append(0)

    return np.array(lim_y)

def get_layer_binary_mask(sample_pred, classes_list, layer='EZ', offset=0):
    binary = (sample_pred == classes_list.index(layer)).astype(int)
    
    if offset > 0:
        size = 1
        for off in range(offset):
            for i in range(size, binary.shape[1], size):
                col = binary[:, i - size:i]
                if np.any(col):
                    place = np.max(np.nonzero(col)[0])
                    binary[place, i - size:i] = 0
    return binary

def get_thickness(binary_image, scale): 
    size = 1
    # print(binary_image.shape)
    thickness = []
    for i in range(size, binary_image.shape[1], size):
        col = binary_image[:, i - size:i]
        if 1 in col:
            thickness.append(np.max(np.where(col)[0]) * scale * 1000 - np.min(np.where(col)[0]) * scale * 1000)

    thickness_nan = np.array(thickness).copy()

    # thickness_nan = thickness_nan[~np.isnan(thickness_nan)]
    # print(thickness_nan.shape, thickness_nan.max(), thickness_nan.min())
    if not np.any(thickness_nan):
        thickness_mean = 0
        thickness_std = 0
    else:
        thickness_mean = np.nanmean(thickness_nan) 
        thickness_std = np.nanstd(thickness_nan) 
    # print(thickness_mean)
    return thickness_mean, thickness_std # micrometers

def get_distance_in_mm(v2_y, v1_y, scale): 
    v2_y = v2_y.astype('float')
    v1_y = v1_y.astype('float')
    v2_y[v2_y == 0] = np.nan
    v1_y[v1_y == 0] = np.nan

    diff_array = np.abs(np.subtract(v2_y, v1_y))
    # diff_array = np.maximum(0, diff_array)
    diff_array = np.multiply(diff_array, scale*1000)
    diff_array[diff_array == 0] = np.nan

    distance_in_mm = diff_array[~np.isnan(diff_array)]

    if not np.any(distance_in_mm):
        distance_in_mm_mean = 0
        distance_in_mm_std = 0
    else:
        distance_in_mm_mean = np.mean(distance_in_mm)
        distance_in_mm_std = np.std(distance_in_mm)
        
    return distance_in_mm_mean, distance_in_mm_std

def get_area(binary_image):
    area_pixels = np.count_nonzero(binary_image == 1)
    return area_pixels

def get_rEZI(ref2, ref1):
    rezi = []
    ref2 = ref2.astype('float')
    ref2[ref2 == 0] = np.nan
    for i in range(ref1.shape[0]):
        relative_diff = (np.abs((ref2[i] - ref1[i])) / ref2[i])
        rezi.append(relative_diff)
    
    rezi_nan = np.array(rezi).copy()

    rezi_nan = rezi_nan[~np.isnan(rezi_nan)]

    if not np.any(rezi_nan):
        rezi_mean = 0
        rezi_std = 0
    else:
        rezi_mean = np.mean(rezi_nan)
        rezi_std = np.std(rezi_nan)

    return rezi_mean * 100, rezi_std * 100

def get_total_variation(segmentation, beta):
    y1 = segmentation / 255.
    vari = 0.0
    local = 0.0
    locales = []
    for k in range(0, y1.shape[1], 2 * beta):
        sample = y1[:, k:k + 2 * beta]
        for j in range(sample.shape[1]):
            vari = np.abs(sample[1, j] - sample[0, j])
            for i in range(2, sample.shape[0]):
                dif = np.abs(sample[i, j] - sample[i - 1, j])
                vari += dif
            local = vari / sample.shape[0]
        locales.append(local)
    locales = np.array(locales)
    # print('TV beta: ',locales.shape)
    locales_nan = np.array(locales).copy()

    locales_nan = locales_nan[~np.isnan(locales_nan)]

    if not np.any(locales_nan):
        tv_mean = 0
        tv_std = 0
    else:
        tv_mean = np.mean(locales_nan)
        tv_std = np.std(locales_nan)
    return locales * 100, tv_mean * 100, tv_std * 100

def gray_gamma(img, gamma):
    gray = img / 255.
    out = np.array(gray ** gamma)
    out = 255*out
    return out.astype('uint8')

def tv_denoising(img, alpha):
    if alpha is not None:
        gray = img / 255.
        out = denoise_tv_chambolle(gray, weight=alpha)
        out = out * 255
    else:
        out = img
    return out.astype('uint8')

def denoising_1D_TV(Y, lamda):
    N = len(Y)
    X = np.zeros(N)

    k, k0, kz, kf = 0, 0, 0, 0
    vmin = Y[0] - lamda
    vmax = Y[0] + lamda
    umin = lamda
    umax = -lamda

    while k < N:
        
        if k == N - 1:
            X[k] = vmin + umin
            break
        
        if Y[k + 1] < vmin - lamda - umin:
            for i in range(k0, kf + 1):
                X[i] = vmin
            k, k0, kz, kf = kf + 1, kf + 1, kf + 1, kf + 1
            vmin = Y[k]
            vmax = Y[k] + 2 * lamda
            umin = lamda
            umax = -lamda
            
        elif Y[k + 1] > vmax + lamda - umax:
            for i in range(k0, kz + 1):
                X[i] = vmax
            k, k0, kz, kf = kz + 1, kz + 1, kz + 1, kz + 1
            vmin = Y[k] - 2 * lamda
            vmax = Y[k]
            umin = lamda
            umax = -lamda
            
        else:
            k += 1
            umin = umin + Y[k] - vmin
            umax = umax + Y[k] - vmax
            if umin >= lamda:
                vmin = vmin + (umin - lamda) * 1.0 / (k - k0 + 1)
                umin = lamda
                kf = k
            if umax <= -lamda:
                vmax = vmax + (umax + lamda) * 1.0 / (k - k0 + 1)
                umax = -lamda
                kz = k
                
        if k == N - 1:
            if umin < 0:
                for i in range(k0, kf + 1):
                    X[i] = vmin
                k, k0, kf = kf + 1, kf + 1, kf + 1
                vmin = Y[k]
                umin = lamda
                umax = Y[k] + lamda - vmax
                
            elif umax > 0:
                for i in range(k0, kz + 1):
                    X[i] = vmax
                k, k0, kz = kz + 1, kz + 1, kz + 1
                vmax = Y[k]
                umax = -lamda
                umin = Y[k] - lamda - vmin
                
            else:
                for i in range(k0, N):
                    X[i] = vmin + umin * 1.0 / (k - k0 + 1)
                break

    return X
