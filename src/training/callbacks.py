
import torch
import matplotlib

from matplotlib import cm
from .dataset import MEAN, STD
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter(SummaryWriter):

    def __init__(self, name_dir):
        self.writer = SummaryWriter(log_dir=name_dir)

    def scalar_epoch(self, scalar_train, scalar_val, step, scalar_name):
        scalar_str = {'Train'+'/'+scalar_name: scalar_train, 'Val'+'/'+scalar_name: scalar_val}
        self.writer.add_scalars(scalar_name, scalar_str, step)

    def metric_iter(self, metric, step, stage, metric_name):
        self.writer.add_scalar(stage + '/' + metric_name, metric, step)

    def loss_iter(self, loss, step, stage: str):
        self.writer.add_scalar(stage + '/Loss', loss, step)

    def learning_rate(self, lr_, step):
        self.writer.add_scalar("lr", lr_, step)

    def save_graph(self, model, loader):
        self.writer.add_graph(model, loader)

    def save_text(self, tag, text_string):
        self.writer.add_text(tag=tag, text_string=text_string)

    def save_images(self, x, y, y_pred, step, device, tag):
        gt = image_tensorboard(y[:3, ...], device)
        if y_pred.shape[1] == 1:
            pred = torch.sigmoid(y_pred[:3, ...])
            pred = torch.round(pred)
        else:
            pred = torch.softmax(y_pred[:3, ...], dim=1)
            pred = torch.argmax(pred, dim=1).unsqueeze(1)
        pred = image_tensorboard(pred, device)
        pred = pred.squeeze(1)
        x1 = denormalize_vis(x[:3, ...])
        self.writer.add_images(f'{tag}/0Input', x1[:3, ...], step, dataformats='NCHW')
        self.writer.add_images(f'{tag}/1True', gt, step, dataformats='NCHW')
        self.writer.add_images(f'{tag}/2Pred', pred, step, dataformats='NCHW')
        # error = torch.abs(gt - pred )
        # self.writer.add_images(f'{tag}/Error', error, step, dataformats='NCHW')

def image_tensorboard(img, device):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=img.max())

    img_rgb = torch.zeros((img.size(0), 3, img.size(2), img.size(3)), dtype=torch.double, device=device)

    for idx in range(1, int(img.max())+1):
        img_rgb[:, 0, :, :] = torch.where(img.squeeze(1) == idx, cm.hsv(norm(idx))[0], img_rgb[:, 0, :, :])
        img_rgb[:, 1, :, :] = torch.where(img.squeeze(1) == idx, cm.hsv(norm(idx))[1], img_rgb[:, 1, :, :])
        img_rgb[:, 2, :, :] = torch.where(img.squeeze(1) == idx, cm.hsv(norm(idx))[2], img_rgb[:, 2, :, :])
    
    return img_rgb

def denormalize_vis(tensor):
    invTrans = transforms.Normalize(mean=-MEAN, std=1/STD)
    return torch.clamp(invTrans(tensor), 0, 1)