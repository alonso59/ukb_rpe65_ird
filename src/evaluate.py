import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from training.metric import MIoU
from training.callbacks import TensorboardWriter
from training.metric import SegmentationMetrics
from training.dataset import data_loaders
from training.loss import WeightedCrossEntropyDice
from training.trainer import evaluation, evaluation_extended
# def evaluation(model, loader, loss_fn, device):
#     loss_acum = 0.0
#     loop = tqdm(loader, ncols=150, ascii=True)
#     iou_fn = MIoU(activation='softmax', ignore_background=False, device=device, average=False)
#     metrics = SegmentationMetrics(ignore_background=False, activation='softmax', average=False)
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (x, y) in enumerate(loop):
#             x = x.type(torch.float).to(device)
#             y = y.type(torch.long).to(device)
#             # forward
#             y_pred = model(x)
#             loss = loss_fn(y_pred, y)
#             # loss function
#             loss_acum += loss.item()
#             # metrics
#             iou = iou_fn(y_pred, y)
#             metrics_value = metrics(y, y_pred)
#     return loss_acum / len(loader), iou, metrics_value

def main():
    base_path = 'logs1/unet_04_06_16_12_31'
    with open(os.path.join(base_path,'experiment_cfg.yaml'), "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    paths = cfg['paths']
    hyper = cfg['hyperparameters']
    # Hyperparameters
    batch_size = hyper['batch_size']
    # Paths
    base = paths['data_base']
    train_imgdir = os.path.join(base, paths['test_imgdir'])
    train_mskdir = os.path.join(base, paths['test_mskdir'])
    test_imgdir = os.path.join(base, paths['test_imgdir'])
    test_mskdir = os.path.join(base, paths['test_mskdir'])

    # General settings

    device = torch.device("cuda")

    train_loader, test_loader = data_loaders(train_imgdir=None,
                                            train_maskdir=None,
                                            val_imgdir=test_imgdir,
                                            val_maskdir=test_mskdir,
                                            batch_size=64,
                                            )
    model = torch.load(os.path.join(base_path, 'checkpoints/model.pth'), map_location='cuda')
    loss_fn = WeightedCrossEntropyDice(device=device, lambda_=0.7, class_weights=[1 for _ in range(9)])
    eval_ = evaluation(model, test_loader, loss_fn, device)
    print('pixel_acc', 'dice', 'precision', 'specificity', 'recall')
    
    print('loss', eval_[0])
    print('mIoU', eval_[1])
    print('BG', 'RNFL', 'GCL', 'INL', 'OPL', 'ONL', 'IS', 'OS', 'RPE')
    print('pixel_acc', eval_[2][0])
    print('pixel_acc', eval_[2][0].mean())
    print('BG', 'RNFL', 'GCL', 'INL', 'OPL', 'ONL', 'IS', 'OS', 'RPE')
    print('dice', eval_[2][1])
    print('dice', eval_[2][1].mean())
    print('BG', 'RNFL', 'GCL', 'INL', 'OPL', 'ONL', 'IS', 'OS', 'RPE')
    print('precision', eval_[2][2])
    print('precision', eval_[2][2].mean())
    print('BG', 'RNFL', 'GCL', 'INL', 'OPL', 'ONL', 'IS', 'OS', 'RPE')
    print('specificity', eval_[2][3])
    print('specificity', eval_[2][3].mean())
    # evaluation_extended(model, test_loader, loss_fn, device)

    # print(f'Loss Eval: {loss_eval:0.4f}')
    # # print(f'IoU Eval: {iou}', iou.mean())
    # # print(f'Acc Eval: {metrics[0]}')
    # print(f'Dice Eval: {metrics[1]}',metrics[1].mean())
    # # print(f'Precision Eval: {metrics[2]}',metrics[2].mean())
    # print(f'Recall Eval: {metrics[3]}',metrics[3].mean())
    # print(f'F1-Score Eval: {metrics[4]}',metrics[4].mean())

if __name__ == '__main__':
    main()