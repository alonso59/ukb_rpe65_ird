import os
import torch
import logging
from torch import nn
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from models.build_model import SegmentationModels
from training.scheduler import CyclicCosineDecayLR
from training.loss import WeightedCrossEntropyDice, LogCoshDice, CrossEntropyLoss
from monai.losses.dice import DiceFocalLoss, DiceLoss

def prepare_paths(base_path, paths):
    """
    Prepare the file paths using the base path and the paths dictionary.

    Args:
        base_path (str): Base path for the file paths.
        paths (dict): Dictionary containing the file paths.

    Returns:
        dict: Dictionary containing the prepared file paths.
    """
    prepared_paths = {}
    for key, value in paths.items():
        prepared_paths[key] = os.path.join(base_path, value)
    return prepared_paths

def prepare_model(device, config, channels, img_sizeh, img_sizew, n_classes, pretrain):
    """
    Prepare the segmentation model.

    Args:
        device (torch.device): Device to use for the model.
        config (dict): Configuration dictionary.
        channels (int): Number of input channels.
        img_sizeh (int): Height of the input images.
        img_sizew (int): Width of the input images.
        n_classes (int): Number of output classes.
        pretrain (bool): Whether to use pre-trained weights.

    Returns:
        nn.Module: Segmentation model.
    """
    models_class = SegmentationModels(device, config_file=config, in_channels=channels,
                                      img_sizeh=img_sizeh, img_sizew=img_sizew,
                                      n_classes=n_classes, pretrain=pretrain)

    model, name_model = models_class.model_building(name_model=config['model_name'])

    models_class.summary(logger=logging.getLogger())

    return model

def prepare_optimizer(model, config):
    """
    Prepare the optimizer based on the configuration.

    Args:
        model (nn.Module): Segmentation model.
        config (dict): Configuration dictionary.

    Returns:
        torch.optim.Optimizer: Optimizer.
    """
    hyper = config['hyperparameters']
    lr = hyper['lr']
    weight_decay = hyper['weight_decay']
    B1 = hyper['b1']
    B2 = hyper['b2']

    if hyper['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))
    elif hyper['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=B1)
    else:
        raise AssertionError('Optimizer not implemented')

    return optimizer

def prepare_loss_function(device, n_classes, hyper):
    """
    Prepare the loss function based on the configuration.

    Args:
        device (torch.device): Device to use for the loss function.
        n_classes (int): Number of output classes.
        hyper (dict): Hyperparameters dictionary.

    Returns:
        nn.Module: Loss function.
    """
    if hyper['loss_fn'] == 'dice_loss':
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    elif hyper['loss_fn'] == 'wce_dice':
        weights = [1 for _ in range(n_classes)]
        loss_fn = WeightedCrossEntropyDice(device=device, lambda_=0.7, class_weights=weights)
    elif hyper['loss_fn'] == 'dice_focal_loss':
        loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, lambda_dice=0.7, lambda_focal=0.3)
    elif hyper['loss_fn'] == 'log_cosh_dice':
        loss_fn = LogCoshDice(device=device)
    elif hyper['loss_fn'] == 'ce':
        loss_fn = CrossEntropyLoss(device=device, weights=[1 for _ in range(n_classes)])
    else:
        raise AssertionError('Loss function not implemented')

    return loss_fn

def prepare_scheduler(optimizer, config):
    """
    Prepare the learning rate scheduler based on the configuration.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        config (dict): Configuration dictionary.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler.
    """
    hyper = config['hyperparameters']
    scheduler_type = hyper['scheduler']['type']

    if scheduler_type == 'step' and scheduler_type is not None:
        scheduler = StepLR(
            optimizer=optimizer, step_size=hyper['scheduler']['step'],
            gamma=hyper['scheduler']['gamma'])
    elif scheduler_type == 'cosine' and scheduler_type is not None:
        scheduler = CyclicCosineDecayLR(optimizer,
                                        init_decay_epochs=150,
                                        min_decay_lr=0.0001,
                                        restart_interval=50,
                                        restart_lr=0.0008)
    elif scheduler_type == 'exponential' and scheduler_type is not None:
        scheduler = ExponentialLR(optimizer, gamma=hyper['scheduler']['gamma'], last_epoch=-1)
    else:
        scheduler = None

    return scheduler