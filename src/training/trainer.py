import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import logging

from tqdm import tqdm
from .callbacks import TensorboardWriter
from training.metric import SegmentationMetrics, MIoU
from sklearn.metrics import classification_report, multilabel_confusion_matrix

class Trainer:
    def __init__(self, model: torch.nn.Module, 
                 optimizer,
                 scheduler,
                 loss_fn: torch.nn.Module, 
                 device: torch.device, 
                 log_path: str, 
                 logger: logging):
        
        self.tb_writer = TensorboardWriter(name_dir=log_path + 'tb/')
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.log_path = log_path
        self.checkpoint_path = log_path + 'checkpoints/'

    def fit(self, train_loader, val_loader, num_epochs, stop_value,save_imgs: bool=True):
        self.__train_iter = 0.0
        self.__val_iter = 0.0
        stop_early = 0

        valid_best_dice = 0.0

        train_metrics_list = []
        valid_metrics_list = []

        train_loss_history = []
        val_loss_history= []
        metric_names = ['Pixel Acc', 'Precision', 'Recall', 'F1_Score', 'DSC', 'mIoU']
        for epoch in range(num_epochs):
            curr_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Epoch: {epoch+1}/{num_epochs} --loss_fn:{self.loss_fn.__name__} --lr:{curr_lr:.3e}")

            train_loss, train_metrics = self.__training(train_loader, save_imgs=save_imgs)
            val_loss, val_metrics = self.__validation(val_loader, save_imgs=save_imgs)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_metrics_list.append(train_metrics)
            valid_metrics_list.append(val_metrics)
            
            self.scheduler.step()
           
            # Saving checkpoint
            if np.array(val_metrics)[4] > valid_best_dice:
                str_print = f"Valid DSC improved from {valid_best_dice:2.5f} to {np.array(val_metrics)[4]:2.5f}.\nSaving checkpoint:{self.checkpoint_path}model.pth"
                best_valid_loss = val_loss
                torch.save(self.model, self.checkpoint_path + f'/model.pth')
                torch.save(self.model.state_dict(), self.checkpoint_path + f'/weights.pth')
                stop_early = 0
                valid_best_dice = np.array(val_metrics)[4]
            else:
                stop_early += 1
                str_print = f"Valid DSC not improved: {valid_best_dice:2.5f}, Val. Loss: {best_valid_loss:2.5f}, ESC: {stop_early}/{stop_value}\nCheckpoint path:{self.checkpoint_path}"
            
            # Reporter
            self.tb_writer.learning_rate(curr_lr, epoch)
            self.tb_writer.scalar_epoch(
                scalar_train=train_loss,
                scalar_val=val_loss,
                step=epoch,
                scalar_name='Loss')
            self.logger.info("================================================================")
            line_new = "{:>15}  {:>15} {:>15}".format("Metric", "Train results", "Val results")
            self.logger.info(line_new)
            self.logger.info("================================================================")
            line_new = "{:>15}  {:>15} {:>15}".format("Loss", f"{train_loss:.5f}", f"{val_loss:0.5f}")
            self.logger.info(line_new)
            self.logger.info("================================================================")
            for idx, name in enumerate(metric_names):
                self.tb_writer.scalar_epoch(scalar_train=np.array(train_metrics)[idx],
                                            scalar_val=np.array(val_metrics)[idx],
                                            step=epoch, 
                                            scalar_name=name)
                line_new = "{:>15}  {:>15} {:>15}".format(f"{name}", f"{np.array(train_metrics)[idx]:.5f}", f"{np.array(val_metrics)[idx]:0.5f}")
                self.logger.info(line_new)
            self.logger.info("================================================================")
            self.logger.info(str_print)
            self.logger.info("================================================================")
            # Stop early
            if stop_value == stop_early:
                self.logger.warning('+++++++++++++++++ Stop training early +++++++++++++')
                break

        # save last checkpoint
        torch.save(self.model, self.checkpoint_path + '/model_last.pth')
        torch.save(self.model.state_dict(), self.checkpoint_path + '/weights_last.pth')
        # Plotting results
        self.__plot_results(np.array(train_loss_history), np.array(val_loss_history), 'Loss', self.log_path)
        for idx, name in enumerate(metric_names):
           self.__plot_results(np.array(train_metrics_list)[:, idx], np.array(valid_metrics_list)[:, idx], metric_names[idx], self.log_path)

    def __training(self, train_loader, save_imgs: bool = True):
        loss_acum = 0.0
        pixel_acc = 0.0
        dice = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        mean_iou = 0.0
        metrics = np.zeros(6)
        len_loader = len(train_loader)
        loop = tqdm(train_loader, ncols=150)
        
        self.model.train()
        for batch_idx, (x, y) in enumerate(loop):
            x = x.type(torch.float).to(self.device)
            y = y.type(torch.long).to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward
            y_pred = self.model(x)
            # loss function
            loss = self.loss_fn(y_pred, y)
            loss_acum += loss.item()
            # backward
            loss.backward()
            self.optimizer.step()
            metrics = self.__segmentation_metrics(y_pred, y)
            pixel_acc += metrics[0]
            precision += metrics[1]
            recall += metrics[2]
            f1_score += metrics[3]
            dice += metrics[4]
            mean_iou += metrics[5]

            self.__tqdm_update(loop, loss, metrics)
            if self.__train_iter % 10 == 0 and save_imgs and self.tb_writer is not None:
                self.tb_writer.save_images(x, y, y_pred, self.__train_iter, self.device, tag='train')
            self.__train_iter += 1

        return loss_acum / len_loader, [pixel_acc / len_loader,
                                        precision / len_loader,
                                        recall / len_loader,
                                        f1_score / len_loader,
                                        dice / len_loader,
                                        mean_iou / len_loader]

    def __validation(self, val_loader, save_imgs: bool = True):
        loss_acum = 0.0
        pixel_acc = 0.0
        dice = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        mean_iou = 0.0
        metrics = np.zeros(6)
        len_loader = len(val_loader)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x = x.type(torch.float).to(self.device)
                y = y.type(torch.long).to(self.device)
                # forward
                y_pred = self.model(x)
                # loss function
                loss = self.loss_fn(y_pred, y)
                loss_acum += loss.item()

                metrics = self.__segmentation_metrics(y_pred, y)
                pixel_acc += metrics[0]
                precision += metrics[1]
                recall += metrics[2]
                f1_score += metrics[3]
                dice += metrics[4]
                mean_iou += metrics[5]            
        
        if self.__val_iter % 10 == 0 and save_imgs and self.tb_writer is not None:
            self.tb_writer.save_images(x, y, y_pred, self.__val_iter, self.device, tag='val')
        self.__val_iter += 1
        return loss_acum / len_loader, [pixel_acc / len_loader,
                                        precision / len_loader,
                                        recall / len_loader,
                                        f1_score / len_loader,
                                        dice / len_loader,
                                        mean_iou / len_loader]

    def __tqdm_update(self, loop: tqdm, loss, metrics):
        loop.set_postfix(
            Loss=loss.item(),
            Pixel_acc=metrics[0],
            Precision=metrics[1],
            Recall=metrics[2],
            f1=metrics[3],
            DSC=metrics[4],
            mIoU=metrics[5]
        )

    def __segmentation_metrics(self, y_pred, y):
        iou_fn = MIoU(ignore_background=False, activation='softmax', device=self.device)
        metrics = SegmentationMetrics(ignore_background=False, activation='softmax', average=True)
        # metrics
        iou = iou_fn(y_pred, y)
        metrics_value = metrics(y, y_pred)

        pixel_acc = metrics_value[0].item()
        dice = metrics_value[1].item()
        precision = metrics_value[2].item()
        recall = metrics_value[3].item()
        f1_score = metrics_value[4].item()
        mean_iou = iou.mean().item()
        
        return [pixel_acc, precision, recall, f1_score, dice, mean_iou]
    
    def __plot_results(self, train, val, name, checkpoint_path):
        # Plot training & validation accuracy values
        plt.figure()
        plt.plot(train)
        plt.plot(val, linestyle='--')
        plt.title(name)
        plt.grid(color='lightgray', linestyle='-', linewidth=2)
        plt.ylabel(name)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='best')
        plt.savefig(checkpoint_path + name + '.png')

def evaluation(model, loader, loss_fn, device):
    loss_acum = 0.0
    iou_fn = MIoU(activation='softmax', ignore_background=False, device=device)
    metrics = SegmentationMetrics(ignore_background=False, activation='softmax', average=False)
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.type(torch.float).to(device)
            y = y.type(torch.long).to(device)
            # forward
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            # loss function
            loss_acum += loss.item()
            # metrics
            iou = iou_fn(y_pred, y)
            metrics_value = metrics(y, y_pred)
    return loss_acum / len(loader), iou.mean(), metrics_value
    
def evaluation_extended(model, loader, loss_fn, device):
    loss_acum = 0.0
    loop = tqdm(loader, ncols=150, ascii=False)
    iou_fn = MIoU(activation='softmax', ignore_background=False, device=device)
    metrics = SegmentationMetrics(ignore_background=False, activation='softmax', average=False)
    model.eval()
    pixel_acc_list = []
    dice_list = []
    precision_list = []
    recall_list = []
    iou_list = []
    y_preds = []
    y_trues = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x = x.type(torch.float).to(device)
            y = y.type(torch.long).to(device)
            # forward
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            # Metrics
            iou = iou_fn(y_pred, y)
            metrics_value = metrics(y, y_pred)
            loss_acum += loss.item()
            iou_list.append(iou)
            pixel_acc_list.append(metrics_value[0])
            dice_list.append(metrics_value[1])
            precision_list.append(metrics_value[2])
            recall_list.append(metrics_value[3])
            y_pred1 = F.softmax(y_pred, dim=1)
            y_pred1 = torch.argmax(y_pred1, dim=1)
            y_pred1 = torch.flatten(y_pred1).detach().cpu().numpy()
            y1 = torch.flatten(y).detach().cpu().numpy()
            y_preds.append(list(y_pred1))
            y_trues.append(list(y1))

    y_preds = np.array(y_preds).reshape(-1)
    y_trues = np.array(y_trues).reshape(-1)
    target_names = np.array(['BG', 'EZ', 'OPL', 'ELM', 'BM'])
    print(classification_report(y_trues, y_preds, target_names=target_names))
    # cf_mtx = multilabel_confusion_matrix(y_trues, y_preds)
    # print(cf_mtx)
    # return loss_acum/len(loader), iou_list, pixel_acc_list, dice_list, precision_list, recall_list

def eval_loss(model, loader, loss_fn, device):
    loss_acum = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.type(torch.float).to(device)
            y = y.type(torch.long).to(device)
            # forward
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            # loss function
            loss_acum += loss.item()
    return loss_acum / len(loader)

    
