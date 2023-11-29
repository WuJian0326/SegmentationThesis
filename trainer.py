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
from config import args
import math
# from memory_profiler import profile

def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1)


import torch

def compute_miou(pred, label):
    class_IoU = []
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

def compute_miou1(pred, label, num_classes):
    class_IoU = []
    miou = 0
    for c in range(num_classes):
        pred_inst = (pred == c).float() 
        label_inst = (label == c).float()

        intersection = torch.logical_and(pred_inst, label_inst)
        union = torch.logical_or(pred_inst, label_inst)

        iou = torch.sum(intersection) / torch.sum(union)
        class_IoU.append(iou.item())
        miou += iou

    miou /= num_classes

    return miou, class_IoU


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

def compute_dice1(pred, label, num_classes):
    class_dice = []
    mdice = 0
    
    for c in range(num_classes):
        pred_inst = (pred == c).float()
        label_inst = (label == c).float()
        
        intersection = torch.logical_and(pred_inst, label_inst)
        dice = 2 * torch.sum(intersection) / (torch.sum(pred_inst) + torch.sum(label_inst))
        class_dice.append(dice.item())
        mdice += dice
        
    mdice /= num_classes
    
    return mdice, class_dice

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
                # print(p1.shape)
                # print(label.shape)
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


class semi_trainer():
    def __init__(self, train_ds, val_ds, model1, model2, optimizer1, optimizer2, scheduler1, scheduler2,
                 criterion1, criterion2, epochs=500,best_acc=None, num_class = 2,trainflow = 2, consistency_weight = 0.1):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model1 = model1
        self.model2 = model2
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_loss1 = best_acc
        self.best_loss2 = best_acc
        self.scaler1 = torch.cuda.amp.GradScaler()
        self.scaler2 = torch.cuda.amp.GradScaler()
        self.num_class = num_class
        self.trainflow = trainflow
        self.consistency_weight = consistency_weight
        self.iters = 0

    def training(self):
        for idx in range(self.epochs):
            self.train_epoch(idx)
            if ((idx+1) % self.trainflow == 0):
                self.validate(idx)
        return self.model1



    def train_epoch(self,epo):

        #
        torch.set_grad_enabled(True)
        self.model1.train()
        self.model2.train()
        total_loss1 = 0


        total_loss2 = 0


        TrainLoader = tqdm(self.train_ds)




        for idx, (img, label, img_path) in enumerate(TrainLoader):
            max_iter = len(self.train_ds) * self.epochs

            volume_batch, label_batch = img.half(), label.unsqueeze(1).float()
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # volume_batch.requires_grad = True
            self.iters += 1

            self.optimizer1.zero_grad(set_to_none=True)
            self.optimizer2.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                output1 = self.model1(volume_batch)
                output2 = self.model2(volume_batch)
                # print(output1[:args.labeled_bs].shape)
                # print(label_batch[:args.labeled_bs].shape)

                output1_softmax = torch.softmax(output1, dim=1)
                output2_softmax = torch.softmax(output2, dim=1)
                consistency_weight = self.consistency_weight * math.exp(-5 * (1 - self.iters / max_iter) ** 2)

                loss1 = 0.3 * (self.criterion1(output1[:args.labeled_bs], label_batch[:args.labeled_bs]) + 0.7 * self.criterion2(
                    output1[:args.labeled_bs], label_batch[:args.labeled_bs], softmax=False))
                loss2 = 0.3 * (self.criterion1(output2[:args.labeled_bs], label_batch[:args.labeled_bs]) + 0.7 * self.criterion2(
                    output2[:args.labeled_bs], label_batch[:args.labeled_bs], softmax=False))
                
                pseudo_outputs1 = torch.argmax(
                    output1_softmax[args.labeled_bs:].detach(), dim=1, keepdim=False)
                pseudo_outputs2 = torch.argmax(
                    output2_softmax[args.labeled_bs:].detach(), dim=1, keepdim=False)
                
                pseudo_supervision1 = self.criterion2(
                    output1_softmax[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
                pseudo_supervision2 = self.criterion2(
                    output2_softmax[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))
                
                model1_loss = loss1 + consistency_weight * pseudo_supervision1
                model2_loss = loss2 + consistency_weight * pseudo_supervision2







            self.scaler1.scale(model1_loss).backward()
            self.scaler1.step(self.optimizer1)
            self.scaler1.update()

            

            self.scaler2.scale(model2_loss).backward()
            self.scaler2.step(self.optimizer2)
            self.scaler2.update()




            TrainLoader.set_description('Epoch ' + str(epo + 1))
            total_loss1 += model1_loss
            total_loss2 += model2_loss
            


        train_loss1 = total_loss1 / len(self.train_ds)
        train_loss2 = total_loss2 / len(self.train_ds)


        self.scheduler1.step()
        self.scheduler2.step()
        l.info(f'Epoch : {epo + 1}, Train_loss1 : {train_loss1}, Train_loss2 : {train_loss2}')


    def validate(self,epo):

        self.model1.eval()
        self.model2.eval()
        total_IoU1 = 0
        total_loss1 = 0
        total_Dice1 = 0
        
        total_IoU2 = 0
        total_loss2 = 0
        total_Dice2 = 0

        with torch.no_grad():
            self.model1.eval()
            self.model2.eval()
            ValLoader = tqdm(self.val_ds)
            for idx, (img, mask, img_path) in enumerate(ValLoader):
                image = img
                mask = mask
                label = mask.long()
                image = image.float()
                image = image.to(self.device) 
                label = label.unsqueeze(1).to(self.device).float()

                p1 = self.model1(image)
                loss_ce1 = self.criterion1(p1, label)
                loss_dice1 = self.criterion2(p1, label, softmax=False)
                loss1 = 0.3 * loss_ce1 + 0.7 * loss_dice1

                outputs1 = p1 

                p2 = self.model2(image)
                loss_ce2 = self.criterion1(p2, label)
                loss_dice2 = self.criterion2(p2, label, softmax=False)
                loss2 = 0.3 * loss_ce2 + 0.7 * loss_dice2

                outputs2 = p2



                outputs1[outputs1 > 0.5] = 1
                outputs1[outputs1 <= 0.5] = 0

                outputs2[outputs2 > 0.5] = 1
                outputs2[outputs2 <= 0.5] = 0

                miou1 = compute_miou(outputs1,label)
                Dice1 = compute_dice(outputs1,label)

                miou2 = compute_miou(outputs2,label)
                Dice2 = compute_dice(outputs2,label)


                total_loss1 += loss1
                total_IoU1 += miou1
                total_Dice1 += Dice1

                total_loss2 += loss2
                total_IoU2 += miou2
                total_Dice2 += Dice2

                

        total_loss1 = total_loss1 / len(self.val_ds)
        total_IoU1 = total_IoU1 / len(self.val_ds)
        total_Dice1 = total_Dice1 / len(self.val_ds)

        total_loss2 = total_loss2 / len(self.val_ds)
        total_IoU2 = total_IoU2 / len(self.val_ds)
        total_Dice2 = total_Dice2 / len(self.val_ds)


        l.info(f'Model1 Validation: Loss : {total_loss1}, mIoU : {total_IoU1}, mDice : {total_Dice1}')
        l.info(f'Model2 Validation: Loss : {total_loss2}, mIoU : {total_IoU2}, mDice : {total_Dice2}')

        self.best_loss1 = save_checkpoint(self.model1, self.best_loss1, total_loss1, epo, model_name= "model1")
        self.best_loss2 = save_checkpoint(self.model2, self.best_loss2, total_loss2, epo, model_name= "model2")