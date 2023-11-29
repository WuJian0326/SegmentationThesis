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
from trainer import compute_miou ,compute_dice, compute_miou1, compute_dice1
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
save_path = "result/Unetr34_semi/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

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

    model = load_checkpoint(model,path='checkpoint/ckpt_model1_49.pth')

    model = model.eval()
    
    





    valid_transform = get_vaild_transform()  # 取得測試影像增強

    testdata = ImageDataLoader(data_path=data_path,
                                 txt_file = test_txt,
                                 transform=valid_transform
                              )
    

    # testdata = FakeDataLoader(data_path="/home/student/Desktop/SegmentationThesis/GAN/FakeImage/",
    #                                 txt_file = "/home/student/Desktop/SegmentationThesis/GAN/FakeImage/GAN64.txt",
    #                                 transform=valid_transform
    #                             )   

    test_loader = DataLoader(testdata, batch_size=1, num_workers=1, pin_memory=True)



    miou1 = 0
    classIOU = [0,0]
    mdice = 0
    classDice = [0,0]
    mDICE1 = 0
    mIoU1 = 0
    dice = 0
    for idx, (img,label, img_path) in enumerate(tqdm(test_loader)):
        image = img.float()
        label = label.long()
        # label1 = lq_mask

        image = image.cuda()

        numpy_array = label.numpy()

        # numpy_array1 = label1.numpy()

        label = label.unsqueeze(1).cuda().float()

        reshaped_array = np.reshape(numpy_array, (224, 224, 1))
        # reshaped_array1 = np.reshape(numpy_array1, (224, 224, 1))

        with torch.no_grad():
            p1= model(image)

            out = p1
            

            out[out > 0.5] = 1
            out[out <= 0.5] = 0
            file_name = img_path[0].split('/')[-1]

            miou1, classIoU = compute_miou1(out,label,2)
            mIoU1 += miou1
            mdice, classDICE = compute_dice1(out,label,2)
            mDICE1 += mdice

            classIOU[0] += classIoU[0]
            classIOU[1] += classIoU[1]

            classDice[0] += classDICE[0]
            classDice[1] += classDICE[1]

            dice += compute_dice(out,label)
            out = out.squeeze(0)
            pred_img = out.cpu().detach().numpy().transpose(1, 2, 0)
            pred_img = pred_img * 255

            image = image.squeeze(0)
            org = image.cpu().detach().numpy().transpose(1, 2, 0) * 255
            image_3chennal = np.zeros((224,224,3))
            org = org[:,:,0]
            pred_img = pred_img[:,:,0]
            image_3chennal[:,:,0] = org
            image_3chennal[:,:,1] = org
            image_3chennal[:,:,2] = org
            pred_img_3chennal = np.zeros((224,224,3))
            pred_img_3chennal[:,:,2] = pred_img

            add_weighted = cv2.addWeighted(image_3chennal, 0.5, pred_img_3chennal, 0.65, 0)
            cv2.imwrite(f'{save_path}{file_name}',add_weighted)

            label = label.squeeze(0)
            label = label.cpu().detach().numpy().transpose(1, 2, 0)
            label = label[:,:,0]
            label = label * 255
            label_3chennal = np.zeros((224,224,3))
            label_3chennal[:,:,2] = label
            img_label = cv2.addWeighted(image_3chennal, 0.5, label_3chennal, 0.65, 0)
            if not os.path.exists(f'./result/gt/'):
                os.makedirs(f'./result/gt/')
            cv2.imwrite(f'./result/gt/{file_name}_label',img_label)

            # img2 = cv2.imread(f'data/milion_UnetPP_Predict_iter1/{file_name}',0)

            # img2 = np.array(Image.open(f'data/milion_UnetPP_Predict_iter1/{file_name}').convert('L'))

            # plt.imshow(img2)
            # plt.show()

            # hq = cv2.imread(f'data/high_quality/{file_name}',0)
            # hq = hq * 255
            # lq = cv2.imread(f'data/low_quality/{file_name}',0)
            # lq = lq * 255
            # org = cv2.imread(f'data/train/{file_name}',0)
            # image = image.squeeze(0)
            # org = image.cpu().detach().numpy().transpose(1, 2, 0)

            # fig, axs = plt.subplots(2, 2)

            # # 在每個子圖上顯示圖像
            # axs[0, 0].imshow(org, cmap='gray')
            # axs[0, 0].set_title('Original')

            # axs[0, 1].imshow(pred_img, cmap='gray')
            # axs[0, 1].set_title('Predicted Image')

            # axs[1, 0].imshow(reshaped_array, cmap='gray')
            # axs[1, 0].set_title('Label')

            # # axs[1, 0].imshow(hq, cmap='gray')
            # # axs[1, 0].set_title('High Quality')

            # # axs[1, 1].imshow(lq, cmap='gray')
            # # axs[1, 1].set_title('Low Quality')

            # # 移除子圖的刻度
            # for ax in axs.flat:
            #     ax.axis('off')

            # # 調整子圖的間距
            # # plt.subplots_adjust(wspace=0, hspace=0)

            # # 保存組合後的圖像
            # # plt.imshow(pred_img, cmap='gray')
            # plt.show()
            # plt.savefig(f'result/{file_name}')
            # plt.close()

    classIOU = [x / len(test_loader) for x in classIOU]
    classDice = [x / len(test_loader) for x in classDice]
    print('miou',mIoU1/len(test_loader))
    print('classIoU',classIoU)
    print('dice',mDICE1/len(test_loader))
    print('classDice',classDice)


    # for idx, (img,mask, img_path) in enumerate(tqdm(test_loader)):
    #     image = img.float()
    #     label = mask.long()
    #     # label1 = lq_mask

    #     image = image.cuda()

    #     # numpy_array = label.numpy()

    #     # numpy_array1 = label1.numpy()

    #     label = label.unsqueeze(1).cuda().float()

    #     # reshaped_array = np.reshape(numpy_array, (224, 224, 1))
    #     # reshaped_array1 = np.reshape(numpy_array1, (224, 224, 1))

    #     with torch.no_grad():
    #         p1= model(image)

    #         out = p1
            

    #         out[out > 0.5] = 1
    #         out[out <= 0.5] = 0
    #         file_name = img_path[0].split('/')[-1]

    #         miou1 += compute_miou(out,label)
    #         out = out.squeeze(0)
    #         pred_img = out.cpu().detach().numpy().transpose(1, 2, 0)
    #         pred_img = pred_img * 255

    #         cv2.imwrite(f'result/Unetr34/{file_name}',pred_img)

    #         # img2 = cv2.imread(f'data/milion_UnetPP_Predict_iter1/{file_name}',0)

    #         # img2 = np.array(Image.open(f'data/milion_UnetPP_Predict_iter1/{file_name}').convert('L'))

    #         # plt.imshow(img2)
    #         # plt.show()

    #         # hq = cv2.imread(f'data/high_quality/{file_name}',0)
    #         # hq = hq * 255
    #         # lq = cv2.imread(f'data/low_quality/{file_name}',0)
    #         # lq = lq * 255
    #         # org = cv2.imread(f'data/train/{file_name}',0)

            

    #         # fig, axs = plt.subplots(2, 2)

    #         # 在每個子圖上顯示圖像
    #         # axs[0, 0].imshow(org, cmap='gray')
    #         # axs[0, 0].set_title('Original')

    #         # axs[0, 1].imshow(pred_img, cmap='gray')
    #         # axs[0, 1].set_title('Predicted Image')

    #         # axs[1, 0].imshow(hq, cmap='gray')
    #         # axs[1, 0].set_title('High Quality')

    #         # axs[1, 1].imshow(lq, cmap='gray')
    #         # axs[1, 1].set_title('Low Quality')

    #         # 移除子圖的刻度
    #         # for ax in axs.flat:
    #         #     ax.axis('off')

    #         # 調整子圖的間距
    #         # plt.subplots_adjust(wspace=0, hspace=0)

    #         # 保存組合後的圖像
    #         # plt.imshow(pred_img, cmap='gray')
    #         # plt.show()
    #         # plt.savefig(f'result/{file_name}')
    #         # plt.close()




if __name__ == '__main__':

    predict()

