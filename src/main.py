import torch
import torch.nn as nn

from training.dataset import *
from training.trainer import *
from utils.initialize import *
from prepare import *

def train(cfg):
    """
    Train the segmentation model.

    Args:
        cfg (dict): Configuration dictionary.
    """
    logger, checkpoint_path, log_path = initialize(cfg)
    paths = cfg['paths']
    hyper = cfg['hyperparameters']
    general = cfg['general']

    # Hyperparameters
    batch_size = hyper['batch_size']
    num_epochs = hyper['num_epochs']
    n_gpus = hyper['n_gpus']

    # Paths
    base_path = paths['data_base']
    prepared_paths = prepare_paths(base_path, paths)

    # General settings
    n_classes = general['n_classes']
    img_sizeh = general['img_sizeh']
    img_sizew = general['img_sizew']
    channels = general['channels']
    device = torch.device(general['device'])

    # Getting loader
    train_loader, val_loader = data_loaders(train_imgdir=prepared_paths['train_imgdir'],
                                            train_maskdir=prepared_paths['train_mskdir'],
                                            val_imgdir=prepared_paths['val_imgdir'],
                                            val_maskdir=prepared_paths['val_mskdir'],
                                            batch_size=batch_size,
                                            num_workers=12)

    logger.info(f'Training items: {len(train_loader) * batch_size}')
    logger.info(f'Validation items: {len(val_loader) * batch_size}')
    logger.info(f'Factor train/val: {len(train_loader) * batch_size / (len(val_loader) * batch_size + len(train_loader) * batch_size)}')

    # Building model
    model = prepare_model(device, cfg, channels, img_sizeh, img_sizew, n_classes, general['pretrain'])

    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=[x for x in range(n_gpus)])

    if general['checkpoint']:
        logger.info('Loading weights...')
        pretrained_path = general['init_weights']
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(pretrained_dict, strict=True)

    # Prepare training
    optimizer = prepare_optimizer(model, cfg)
    loss_fn = prepare_loss_function(device, n_classes, hyper)
    scheduler = prepare_scheduler(optimizer, cfg)

    logger.info('**********************************************************')
    logger.info('**************** Initialization successful ****************')
    logger.info('**********************************************************')
    logger.info('--------------------- Start training ---------------------')

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      scheduler=scheduler, 
                      loss_fn=loss_fn, 
                      device=device, 
                      log_path=log_path, 
                      logger=logger)
    
    trainer.fit(train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                stop_value=int(num_epochs*0.2))
    
    logger.info('-------------------- Finished Train ---------------------')
    logger.info('******************* Start evaluation  *******************')
    model = torch.load(checkpoint_path + 'model.pth')
    eval_result = evaluation(model, val_loader, loss_fn, device)
    logger.info(eval_result)

if __name__ == '__main__':
    cfg = load_config()
    train(cfg)
