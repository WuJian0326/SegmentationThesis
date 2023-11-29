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
from trainer import trainer, semi_trainer
# from torchinfo import summary
from DataLoader import *
from utils import *
import segmentation_models_pytorch as smp
from Dice import DiceLoss
from config import args
# import metric
from torch.optim.lr_scheduler import LambdaLR
from model.model import SwinUnet


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=32'




lr = args.lr
batch_size = args.batch_size
num_epoch = args.epoch
num_worker = args.num_worker
num_class = args.num_class
in_channels = args.in_channels
image_size = args.image_size
trainflow = args.train_flow
data_path = args.data_path
train_txt = args.train_txt
val_txt = args.val_txt
test_txt = args.test_txt
unlabel_txt = args.unlabel_txt
consistency_weight = args.consistency_weight


l.info(f'learning rate : {lr}')
l.info(f'batch size : {batch_size}')
l.info(f'num epoch : {num_epoch}')
l.info(f'trainflow: {trainflow}')


device = "cuda" if torch.cuda.is_available() else "cpu"
pin_memory = True


SMOOTH = 1e-6




def semi_train_model():

    l.info("==========Start semi training=========")
    train_transform = get_transform()  # 取得影像增強方式
    valid_transform = get_vaild_transform()  # 取得測試影像增強
    # 將資料送進dataloader中
    label_data = ImageDataLoader(
                                 data_path=data_path,
                                 txt_file = train_txt,
                                 transform=train_transform
                              )
    
    unlabel_data = FakeDataLoader(
                                data_path="/home/student/Desktop/SegmentationThesis/GAN/FakeImage/",
                                txt_file = unlabel_txt,
                                transform=train_transform
                              )


    val_data = ImageDataLoader(data_path=data_path,
                                 txt_file = val_txt,
                                 transform=valid_transform
                              )

    
       



    labeled_idxs = list(range(0, len(label_data)))
    unlabeled_idxs = list(range(len(label_data), len(label_data) + len(unlabel_data)))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size- args.labeled_bs)

    mix_dataset = torch.utils.data.ConcatDataset([label_data, unlabel_data])

    train_loader = torch.utils.data.DataLoader(
        mix_dataset,
        batch_sampler=batch_sampler,
        num_workers=4)

    
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_worker, pin_memory=True)

    model1 = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7

        in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_class,  # model output channels (number of classes in your dataset)
    ).to(device)

    # model2 = FCT().to(device)

    model2 = SwinUnet(img_size=image_size, in_chans=3, num_classes=num_class
            ).to(device)
    
    # model1 =  load_checkpoint(model1,path='checkpoint/ckpt_model1_49.pth')
    # model2 =  load_checkpoint(model2,path='checkpoint/ckpt_model2_114.pth')


    loss_function１ = nn.CrossEntropyLoss()
    loss_function2 = DiceLoss(n_classes=num_class)


    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)
    

    scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.95)
    scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.95)

    train = semi_trainer(train_loader, val_loader, model1, model2, optimizer1, optimizer2, scheduler1 , scheduler2, loss_function1,loss_function2,
                    epochs=num_epoch, best_acc=None, num_class= num_class, trainflow = trainflow, consistency_weight = consistency_weight)
    # 訓練


    model = train.training()

    l.info("==========finish train=========")



if __name__ == '__main__':

    semi_train_model()


    l.info("==========finish work=========")