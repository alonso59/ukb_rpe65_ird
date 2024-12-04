import matplotlib.pyplot as plt
import torch
import sys
import os
import yaml
import numpy as np

from ray import tune
from ray.tune.suggest.ax import AxSearch
from ax.service.ax_client import AxClient

from training.dataset import loaders
from src.build_model import SegmentationModels
from training.trainer import fit, evaluation, eval_loss
from utils.initialize import initialize as init
from torch.optim.lr_scheduler import StepLR
from training.scheduler import CyclicCosineDecayLR
from monai.losses.dice import DiceFocalLoss, DiceLoss

from training.loss import WeightedCrossEntropyDice, CrossEntropyLoss
from functools import partial

def experiment(config):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(current_dir, '../configs/oct.yaml'), "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    # Hyperparameters
    batch_size = 128
    num_epochs = 200
    lr = config["lr"]
    B1 = config["B1"]
    B2 = config["B2"]
    weight_decay = config["weight_decay"]
    print(lr)
    logger, checkpoint_path, version = init(cfg, bayessian=True)
    paths = cfg['paths']
    hyper = cfg['hyperparameters']
    general = cfg['general']
    # Paths
    base = paths['base']
    train_imgdir = current_dir + os.path.join(base, paths['train_imgdir'])
    train_mskdir = current_dir + os.path.join(base, paths['train_mskdir'])
    val_imgdir = current_dir + os.path.join(base, paths['val_imgdir'])
    val_mskdir = current_dir + os.path.join(base, paths['val_mskdir'])
    pretrain = general['pretrain']
    # General settings
    n_classes = general['n_classes']
    img_sizeh = general['img_sizeh']
    img_sizew = general['img_sizew']
    channels = general['channels']
    name_model = cfg['model_name']
    device = torch.device("cuda")
    # Getting loader
    train_loader, val_loader, data_augmentation = loaders(train_imgdir=train_imgdir,
                                                          train_maskdir=train_mskdir,
                                                          val_imgdir=val_imgdir,
                                                          val_maskdir=val_mskdir,
                                                          batch_size=batch_size,
                                                          num_workers=24,
                                                          pin_memory=True,
                                                          preprocess_input=None,
                                                          )
    # for f in data_augmentation.transforms:
    #     op = json.dumps(f.get_transform_init_args())
    #     logger.info(f'{f.__class__.__name__}, p={f.p}, {op}')
    logger.info(f'Training items: {len(train_loader) * batch_size}')
    logger.info(f'Validation items: {len(val_loader) * batch_size}')
    logger.info(f'Factor train/val: {len(train_loader) * batch_size / (len(val_loader) * batch_size + len(train_loader) * batch_size)}')

    # Building model
    models_class = SegmentationModels(device, config_file=cfg, in_channels=channels,
                                      img_sizeh=img_sizeh, img_sizew=img_sizew,
                                      n_classes=n_classes, pretrain=pretrain)

    model, name_model = models_class.model_building(name_model=name_model)

    models_class.summary(logger=logger)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.info(f'Total_params:{pytorch_total_params}')

    # Prepare training
    if hyper['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))
    elif hyper['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=B1)
    else:
        raise AssertionError('Optimizer not implemented')

    if hyper['loss_fn'] == 'dice_loss':
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    if hyper['loss_fn'] == 'wce_dice':
        loss_fn = WeightedCrossEntropyDice(device=device, lambda_=0.6, class_weights=[1, 1, 1, 1, 1]) #[0.21011505, 22.14241546, 14.21335034, 23.14514484, 12.19832582]
    if hyper['loss_fn'] == 'dice_focal_loss':
        loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, lambda_dice=0.7, lambda_focal=0.3)
    if hyper['loss_fn'] == 'ce':
        loss_fn = CrossEntropyLoss(device=device, weights=[1 for _ in range(n_classes)])


    scheduler = StepLR(
        optimizer=optimizer, step_size=cfg['hyperparameters']['scheduler']['step'],
        gamma=cfg['hyperparameters']['scheduler']['gamma'])


    # """ Trainer """
    logger.info('**********************************************************')
    logger.info('**************** Initialization sucessful ****************')
    logger.info('**********************************************************')
    logger.info('--------------------- Start training ---------------------')
    fit(num_epochs=num_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            checkpoint_path=checkpoint_path,
            scheduler=scheduler,
            name_model=name_model,
            callback_stop_value=int(num_epochs * 0.15),
            tb_dir=version,
            logger=logger
            )
    logger.info('-------------------- Finished Train ---------------------')
    logger.info('******************* Start evaluation  *******************')
    load_best_model = torch.load(checkpoint_path + 'model.pth')
    eval = evaluation(load_best_model, val_loader, loss_fn, device)
    logger.info(eval)
    logger.info('************************* Report to Ray  **************************')
    tune.report(loss_eval=eval_loss(load_best_model, val_loader, loss_fn, device))


def bayessian():
    ax = AxClient(enforce_sequential_optimization=False)
    
    ax.create_experiment(name="swin_experiment",
                         parameters=[
                            {"name": "lr", "type": "range", "bounds": [5e-4, 1e-3], "log_scale": True},
                            {"name": "weight_decay", "type": "range", "bounds": [1e-5, 1e-3]},
                            {"name": "B1", "type": "range", "bounds": [0.7, 0.9]},
                            {"name": "B2", "type": "range", "bounds": [0.7, 0.999]}
                         ],
                         objective_name="loss_eval",
                         minimize=True
                        )
    # Set up AxSearcher in RayTune
    algo = AxSearch(ax_client=ax)
    # Wrap AxSearcher in a concurrently limiter, to ensure that Bayesian optimization receives the
    # data for completed trials before creating more trials
    algo = tune.suggest.ConcurrencyLimiter(algo, max_concurrent=2)
    tune.run(experiment,
            num_samples=20,
            search_alg=algo,
            resources_per_trial={"cpu": 8, "gpu": 4},
            verbose=0,  # Set this level to 1 to see status updates and to 2 to also see trial results.
            # To use GPU, specify: resources_per_trial={"gpu": 1}.
            )
    best_parameters, values = ax.get_best_parameters()
    means, covariances = values
    print(best_parameters)
    print(means)
    print(covariances)
if __name__ == '__main__':
    bayessian()