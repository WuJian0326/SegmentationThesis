import cv2
import os
from model.FCT import FCT
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from trainer import trainer
# from torchinfo import summary
from DataLoader import *
from utils import *
import segmentation_models_pytorch as smp
from Dice import DiceLoss
import argparse
# import metric
from torch.optim.lr_scheduler import LambdaLR
from model.model import SwinUnet


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=32'

parser = argparse.ArgumentParser()
parser.add_argument("--train",default=True,help="learning rate",action="store_true")
parser.add_argument("--predict",default=False,help="learning rate",action="store_true")
parser.add_argument("--lr",default=1e-3,help="learning rate",type = float)
parser.add_argument("-b","--batch_size",default=4,help="batch_size",type = int)
parser.add_argument("-e","--epoch",default=300,help="num_epoch",type = int)
parser.add_argument("-worker","--num_worker",default=4,help="num_worker",type = int)
parser.add_argument("-class","--num_class",default=1,help="num_class",type = int)
parser.add_argument("-c","--in_channels",default=1,help="in_channels",type = int)
parser.add_argument("-size","--image_size",default=224,help="image_size",type = int)
parser.add_argument("-flow","--train_flow",default=10,help="image_size",type = int)
parser.add_argument("-data_path","--data_path",default="/home/student/Desktop/SegmentationThesis/data/microglia/",help="data path",type = str)
parser.add_argument("-train_txt","--train_txt",default="/home/student/Desktop/SegmentationThesis/data/train.txt",help="train_txt",type = str)
parser.add_argument("-val_txt","--val_txt",default="/home/student/Desktop/SegmentationThesis/data/validation.txt",help="val_txt",type = str)
parser.add_argument("-test_txt","--test_txt",default="/home/student/Desktop/SegmentationThesis/data/test.txt",help="test_txt",type = str)

args = parser.parse_args()


lr = args.lr
batch_size = args.batch_size
num_epoch = args.epoch
num_worker = args.num_worker
num_class = args.num_class
in_channels = args.in_channels
image_size = args.image_size
trainflow = args.train_flow
doTrain  = args.train
doPredict = args.predict
data_path = args.data_path
train_txt = args.train_txt
val_txt = args.val_txt
test_txt = args.test_txt


l.info(f'learning rate : {lr}')
l.info(f'batch size : {batch_size}')
l.info(f'num epoch : {num_epoch}')
l.info(f'trainflow: {trainflow}')


device = "cuda" if torch.cuda.is_available() else "cpu"
pin_memory = True


SMOOTH = 1e-6










def train_model():

    l.info("==========Start training=========")
    train_transform = get_transform()  # 取得影像增強方式
    valid_transform = get_vaild_transform()  # 取得測試影像增強
    # 將資料送進dataloader中
    train_data = ImageDataLoader(
                                 data_path=data_path,
                                 txt_file = train_txt,
                                 transform=train_transform
                              )
    



    val_data = ImageDataLoader(data_path=data_path,
                                 txt_file = val_txt,
                                 transform=valid_transform
                              )

    
    
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_worker, pin_memory=True)

    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_worker, pin_memory=True)

    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7

        in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_class,  # model output channels (number of classes in your dataset)
    ).to(device)

    # model = FCT().to(device)

    # model = SwinUnet(img_size=image_size, in_chans=3, num_classes=num_class
    #         ).to(device)


    loss_function１ = nn.CrossEntropyLoss()
    loss_function2 = DiceLoss(n_classes=num_class)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train = trainer(train_loader, val_loader, model, optimizer, scheduler, loss_function1,loss_function2,
                    epochs=num_epoch, best_acc=None, num_class= num_class, trainflow = trainflow)
    # 訓練


    model = train.training()

    l.info("==========finish train=========")



if __name__ == '__main__':

    train_model()


    l.info("==========finish work=========")