import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from utils.summary import summary
from .networks.unet import UNet
from .networks.R2U_Net import R2U_Net
from .networks.MGU_Net import MGUNet_1
from .networks.ce_net import CE_Net_OCT
from .networks.swin_unet import SwinUnet
from monai.networks.nets.basic_unet import BasicUNet
from monai.networks.nets.unet import UNet as UNet_monai
from monai.networks.nets.swin_unetr import SwinUNETR as SwinUNETR_monai
from relaynet_pytorch.relay_net import ReLayNet

class SegmentationModels(nn.Module):
    '''
    SegmentationModels class is used to build the segmentation model based on the configuration file.
    
    Args:
    - device: device to run the model on
    - in_channels: number of input channels
    - img_sizeh: height of the input image
    - img_sizew: width of the input image
    - config_file: yaml configuration file containing the model architecture
    - n_classes: number of classes in the dataset
    - pretrain: whether to use pre-trained weights
    - pretrained_path: path to the pre-trained weights
    
    Returns:
    - model: segmentation model
    - name: name of the model
    
    
    '''
    def __init__(self, device, in_channels, img_sizeh, img_sizew, config_file, n_classes=1, pretrain=True, pretrained_path=None) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.img_sizeh = img_sizeh
        self.img_sizew = img_sizew
        self.n_classes = n_classes
        self.pretrain = pretrain
        self.config_file = config_file
        self.pretrained_path = pretrained_path

    def model_building(self, name_model='unet'):
        '''
        model_building function is used to build the segmentation model based on the configuration file.
        It supports the following models:
            - UNet
            - UNet++
            - SwinUNet
            - SwinUNet_Custom
            - PSPNet_imagenet
            - UNet_monai
            - SwinUNETR
            - BasicUNet_monai
            - CE_Net
            - MGU_Net
            - ReLayNet
            - R2U_Net
            - UNet_imagenet
        
        Args:
        - name_model: name of the model to build
        
        Returns:
        - model: segmentation model
        - name: name of the model
        
        '''
        if name_model == 'unet':
            feature_start = self.config_file['unet_architecutre']['feature_start']
            layers = self.config_file['unet_architecutre']['layers']
            bilinear = self.config_file['unet_architecutre']['bilinear']
            dropout = self.config_file['unet_architecutre']['dropout']
            kernel_size = self.config_file['unet_architecutre']['kernel_size']
            stride = self.config_file['unet_architecutre']['stride']
            padding = self.config_file['unet_architecutre']['padding']
            self.model, name = self.UNet(feature_start, layers, bilinear, dropout, kernel_size, stride, padding)

        if name_model == 'swin_unet':
            self.model, name = self.swin_unet()

        if name_model == 'swin_unet_custom':
            embed_dim = self.config_file['swin_unet_custom_architecture']['embed_dim']
            depths = self.config_file['swin_unet_custom_architecture']['depths']
            num_heads = self.config_file['swin_unet_custom_architecture']['num_heads']
            window_size = self.config_file['swin_unet_custom_architecture']['window_size']
            drop_path_rate = self.config_file['swin_unet_custom_architecture']['drop_path_rate']
            self.model, name = self.SwinUnet_Custom(embed_dim, depths, num_heads, window_size, drop_path_rate)

        if name_model == 'PSPNet_imagenet':
            encoder_name = self.config_file['unet_encoder']
            weights = None
            if self.pretrain:
                weights = "imagenet"
            self.model, name = self.Unet_backbone(encoder_name=encoder_name, encoder_weights=weights)

        if name_model == 'unet_monai':
            feature_start = self.config_file['unet_monai_architecture']['feature_start']
            layers = self.config_file['unet_monai_architecture']['layers']
            dropout = self.config_file['unet_monai_architecture']['dropout']
            kernel_size = self.config_file['unet_monai_architecture']['kernel_size']
            num_res_units = self.config_file['unet_monai_architecture']['num_res_units']
            norm = self.config_file['unet_monai_architecture']['norm']
            self.model, name = self.unet_monai(features_start=feature_start, num_layers=layers, norm=norm,
                                               num_res_units=num_res_units, kernel_size=kernel_size, dropout=dropout)

        if name_model == 'swinUNETR':
            self.model = SwinUNETR_monai(img_size=(self.img_sizeh, self.img_sizew), in_channels=self.in_channels, 
                                         out_channels=self.n_classes, use_checkpoint=True, spatial_dims=2, norm_name='batch')
            self.model.to(self.device)
            name = 'SwinUNETR'

        if name_model == 'basicunet_monai':
            self.model = BasicUNet(spatial_dims=2, features=(16, 32, 64, 128, 64, 32),
                                   out_channels=self.n_classes, norm='batch', in_channels=self.in_channels).to(self.device)
            name = 'basicunet_monai'

        if name_model == 'ce_net':
            self.model = CE_Net_OCT(num_classes=self.n_classes).to(self.device)
            name = 'ce_net'

        if name_model == 'MGU_Net':
            self.model = MGUNet_1(in_channels=self.in_channels, n_classes=self.n_classes).to(self.device)
            name = 'MGU_Net'

        if name_model == 'relaynet':
            self.model = ReLayNet(params ={
                                    'num_channels':1,
                                    'num_filters':64,
                                    'kernel_h':7,
                                    'kernel_c':1,
                                    'kernel_w':3,
                                    'stride_conv':1,
                                    'pool':2,
                                    'stride_pool':2,
                                    'num_class':self.n_classes
                                }).to(self.device)
            name = 'relaynet'

        if name_model == 'R2U_Net':
            self.model = R2U_Net(img_ch=self.in_channels, output_ch=self.n_classes, t=2).to(self.device)
            name = 'R2U_Net'
        if name_model == 'unet_imagenet':
            encoder_name = self.config_file['unet_encoder']
            weights = None
            if self.pretrain:
                weights = "imagenet"
            self.model, name = self.Unet_backbone(encoder_name=encoder_name, encoder_weights=weights)
        return self.model, name

    def summary(self, logger=None):
        '''
        summary function is used to print the model summary

        Args:
        - logger: logger object to log the summary

        Returns:
        - None
        
        '''
        summary(self.model, input_size=(self.in_channels, self.img_sizeh, self.img_sizew), batch_size=1, logger=logger)

    def UNet(self, feature_start=16, layers=4, bilinear=False, dropout=0.0, kernel_size=3, stride=1, padding=1):
        '''
        UNet function is used to build the UNet model
        
        Args:
        - feature_start: number of features in the first layer
        - layers: number of layers in the model
        - bilinear: whether to use bilinear interpolation (False ConvTranspose2d, True bilinear interpolation)
        - dropout: dropout rate
        - kernel_size: kernel size
        - stride: stride
        - padding: padding

        Returns:
        - model: UNet model
        - name: name of the model
            
        '''
        model = UNet(
            num_classes=self.n_classes,
            input_channels=self.in_channels,
            num_layers=layers,
            features_start=feature_start,
            bilinear=bilinear,
            dp=dropout,
            kernel_size=(kernel_size, kernel_size),
            padding=padding,
            stride=stride
        ).to(self.device)
        if self.pretrain:
            pass
        return model, model.__name__

    def unet_monai(self, features_start=32, num_layers=4, num_res_units=0, kernel_size=3, dropout=0.0, norm='batch'):
        channels = [features_start]
        strides = []
        for _ in range(1, num_layers):
            channels.append(channels[-1] * 2)
            strides.append(2)

        model = UNet_monai(
            spatial_dims=2,
            in_channels=self.in_channels,
            out_channels=self.n_classes,
            channels=channels,
            strides=strides,
            act='leakyrelu',
            num_res_units=num_res_units,
            kernel_size=kernel_size,
            norm=norm,
            dropout=dropout,
        ).to(self.device)
        name = 'UNet_monai'
        return model, name

    def swin_unet(self,
                  embed_dim=96,
                  depths=[2, 2, 6, 2],
                  num_heads=[3, 6, 12, 24],
                  window_size=8,
                  drop_path_rate=0.1,
                  ):

        model = SwinUnet(
            img_size=self.img_sizeh,
            num_classes=self.n_classes,
            zero_head=False,
            patch_size=4,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
        ).to(self.device)

        if self.pretrain:
            model.state_dict()
            model.load_from("pretrained/swin_tiny_patch4_window7_224.pth", self.device)
        return model, model.__name__

    def SwinUnet_Custom(self,
                        embed_dim=24,
                        depths=[2, 2, 2, 2],
                        num_heads=[2, 2, 2, 2],
                        window_size=7,
                        drop_path_rate=0.1,
                        ):

        model = SwinUnet(
            img_size=self.img_sizeh,
            num_classes=self.n_classes,
            zero_head=False,
            patch_size=4,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
        ).to(self.device)

        return model, model.__name__

    def Unet_backbone(self, encoder_name="resnet18", encoder_weights="imagenet"):
        model = smp.Unet(
            encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=self.in_channels,
            classes=self.n_classes,                      # model output channels (number of classes in your dataset)
        ).to(self.device)

        return model, 'unet_imagenet'

    def Unet_plusplus(self, encoder_name="resnet18", encoder_weights="imagenet"):
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
            decoder_channels = (128, 64, 32, 16),
            encoder_depth=4,
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=self.in_channels,
            classes=self.n_classes,                      # model output channels (number of classes in your dataset)
        ).to(self.device)

        return model, 'unet_imagenet'

    """
    you can add your own network here
    .
    .
    .
    """
