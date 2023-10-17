import cv2
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
from PIL import Image

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms , models

# from torchinfo import summary
from DataLoader import *
from utils import *
import segmentation_models_pytorch as smp
from Dice import DiceLoss
import argparse
# import metric
from torch.optim.lr_scheduler import LambdaLR
from trainer import compute_miou 
from model.model import SwinUnet
import argparse 
from model.FCT import FCT

parser = argparse.ArgumentParser()
parser.add_argument("-data_path","--data_path",default="/home/student/Desktop/SegmentationThesis/data/microglia/",help="data path",type = str)
parser.add_argument("-train_txt","--train_txt",default="/home/student/Desktop/SegmentationThesis/data/train.txt",help="train_txt",type = str)
parser.add_argument("-val_txt","--val_txt",default="/home/student/Desktop/SegmentationThesis/data/validation.txt",help="val_txt",type = str)
parser.add_argument("-test_txt","--test_txt",default="/home/student/Desktop/SegmentationThesis/data/test.txt",help="test_txt",type = str)
args = parser.parse_args()

data_path = args.data_path
test_txt = args.test_txt


device = "cuda" if torch.cuda.is_available() else "cpu"

def predict():


    # model = build_model(Predict = True)
    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    ).to(device)
    # model = FCT().to(device)
    # model = SwinUnet(img_size=224, in_chans=3, num_classes=1
    #         ).to(device)

    model = load_checkpoint(model,path='checkpoint/Unetres34.pth')

    model = model.eval()
    
    





    valid_transform = get_vaild_transform()  # 取得測試影像增強

    testdata = ImageDataLoader(data_path=data_path,
                                 txt_file = test_txt,
                                 transform=valid_transform
                              )



    test_loader = DataLoader(testdata, batch_size=1, num_workers=1, pin_memory=True)



    miou1 = 0

    for idx, (img,mask, img_path) in enumerate(tqdm(test_loader)):
        image = img.float()
        label = mask.long()
        # label1 = lq_mask

        image = image.cuda()

        # numpy_array = label.numpy()

        # numpy_array1 = label1.numpy()

        label = label.unsqueeze(1).cuda().float()

        # reshaped_array = np.reshape(numpy_array, (224, 224, 1))
        # reshaped_array1 = np.reshape(numpy_array1, (224, 224, 1))

        with torch.no_grad():
            p1= model(image)

            out = p1
            

            out[out > 0.5] = 1
            out[out <= 0.5] = 0
            file_name = img_path[0].split('/')[-1]

            miou1 += compute_miou(out,label)
            out = out.squeeze(0)
            pred_img = out.cpu().detach().numpy().transpose(1, 2, 0)
            pred_img = pred_img * 255

            cv2.imwrite(f'result/Unetr34/{file_name}',pred_img)

            # img2 = cv2.imread(f'data/milion_UnetPP_Predict_iter1/{file_name}',0)

            # img2 = np.array(Image.open(f'data/milion_UnetPP_Predict_iter1/{file_name}').convert('L'))

            # plt.imshow(img2)
            # plt.show()

            # hq = cv2.imread(f'data/high_quality/{file_name}',0)
            # hq = hq * 255
            # lq = cv2.imread(f'data/low_quality/{file_name}',0)
            # lq = lq * 255
            # org = cv2.imread(f'data/train/{file_name}',0)

            

            # fig, axs = plt.subplots(2, 2)

            # 在每個子圖上顯示圖像
            # axs[0, 0].imshow(org, cmap='gray')
            # axs[0, 0].set_title('Original')

            # axs[0, 1].imshow(pred_img, cmap='gray')
            # axs[0, 1].set_title('Predicted Image')

            # axs[1, 0].imshow(hq, cmap='gray')
            # axs[1, 0].set_title('High Quality')

            # axs[1, 1].imshow(lq, cmap='gray')
            # axs[1, 1].set_title('Low Quality')

            # 移除子圖的刻度
            # for ax in axs.flat:
            #     ax.axis('off')

            # 調整子圖的間距
            # plt.subplots_adjust(wspace=0, hspace=0)

            # 保存組合後的圖像
            # plt.imshow(pred_img, cmap='gray')
            # plt.show()
            # plt.savefig(f'result/{file_name}')
            # plt.close()




if __name__ == '__main__':

    predict()

