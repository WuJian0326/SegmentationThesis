from tqdm import tqdm
from utils import *
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
from time import time as tm
import matplotlib.pyplot as plt
SMOOTH = 1e-6
# from memory_profiler import profile

def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1)


import torch

def compute_miou(pred, label):

    miou = 0
    for i in range(pred.shape[0]):
        pred_inst = pred[i, 0]
        label_inst = label[i, 0]
        
        intersection = torch.logical_and(pred_inst, label_inst)
        union = torch.logical_or(pred_inst, label_inst)
        
        iou = torch.sum(intersection) / torch.sum(union)
        miou += iou
        
    miou /= pred.shape[0]
    
    return miou

def compute_dice(pred, label):

    mdice = 0
    for i in range(pred.shape[0]):
        pred_inst = pred[i, 0]
        label_inst = label[i, 0]
        
        intersection = torch.logical_and(pred_inst, label_inst)
        dice = 2 * torch.sum(intersection) / (torch.sum(pred_inst) + torch.sum(label_inst))
        mdice += dice
        
    mdice /= pred.shape[0]
    
    return mdice


class trainer():
    def __init__(self, train_ds, val_ds, model, optimizer, scheduler,
                 criterion1, criterion2, epochs=500,best_acc=None, num_class = 2,trainflow = 2):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_loss = best_acc
        self.scaler = torch.cuda.amp.GradScaler()
        self.num_class = num_class
        self.trainflow = trainflow

    def training(self):
        for idx in range(self.epochs):
            self.train_epoch(idx)
            if ((idx+1) % self.trainflow == 0):
                self.validate(idx)
        return self.model



    def train_epoch(self,epo):

        #
        torch.set_grad_enabled(True)
        self.model.train()
        total_loss = 0
        total_IoU = 0
        total_Dice = 0

        TrainLoader = tqdm(self.train_ds)


        for idx, (img, high_mask, img_path) in enumerate(TrainLoader):
            

            image = img
            hq_mask = high_mask 
            label = hq_mask.long()
            image = image.half()
            image = image.to(self.device) 
            label = label.unsqueeze(1).to(self.device).float()



            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                

                p1 = self.model(image)
                loss_ce1 = self.criterion1(p1, label)
                loss_dice1 = self.criterion2(p1, label, softmax=False)
                loss1 = 0.3 * loss_ce1 + 0.7*loss_dice1
                loss = loss1 
                outputs = p1 




            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            TrainLoader.set_description('Epoch ' + str(epo + 1))
            total_loss += loss
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0

            total_IoU += compute_miou(outputs,label)
            total_Dice += compute_dice(outputs,label)


        train_loss = total_loss / len(self.train_ds)
        mean_IoU = total_IoU / len(self.train_ds)
        mean_Dice = total_Dice / len(self.train_ds)

        self.scheduler.step()
        l.info(f'Epoch : {epo + 1}, Train_loss : {train_loss}, Mean_ioU: {mean_IoU}, Mean_Dice: {mean_Dice}')


    def validate(self,epo):

        self.model.eval()
        total_IoU = 0
        total_loss = 0
        total_Dice = 0

        with torch.no_grad():
            self.model.eval()
            ValLoader = tqdm(self.val_ds)
            for idx, (img, mask, img_path) in enumerate(ValLoader):
                image = img
                mask = mask
                label = mask.long()
                image = image.float()
                image = image.to(self.device) 
                label = label.unsqueeze(1).to(self.device).float()

                p1 = self.model(image)
                loss_ce1 = self.criterion1(p1, label)
                loss_dice1 = self.criterion2(p1, label, softmax=False)
                loss1 = 0.3 * loss_ce1 + 0.7 * loss_dice1
                loss = loss1 
                outputs = p1 





                outputs[outputs > 0.5] = 1
                outputs[outputs <= 0.5] = 0

                miou1 = compute_miou(outputs,label)
                Dice = compute_dice(outputs,label)


                total_loss += loss
                total_IoU += miou1
                total_Dice += Dice
                

        total_loss = total_loss / len(self.val_ds)
        total_IoU = total_IoU / len(self.val_ds)
        total_Dice = total_Dice / len(self.val_ds)

        l.info(f'Validation: Loss : {total_loss}, mIoU : {total_IoU}, mDice : {total_Dice}')
        self.best_loss = save_checkpoint(self.model, self.best_loss, total_loss, epo)
