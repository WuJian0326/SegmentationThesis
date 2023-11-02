import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from tqdm import trange, tqdm
from time import time
import random
import torch
from skimage.transform import rotate


def load_file_list(file_path, data_path = 'data/'):
    file_list = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # 移除開頭和結尾的空格或換行符號

            file_list.append(line)

    return file_list



def get_transform():

    transform = A.Compose([
        A.Resize(224,224),
        A.OneOf([

            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.3),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4,
            #                    p=0.3),
            A.RandomRotate90(),
        ], p=0.3),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2(),
        
    ])

    return transform

def get_vaild_transform():
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2(),
        
    ])
    
    return transform

def generate_cell_kernel(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # 連通組件分析
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(closing)

    # 設定面積和距離閾值
    min_area = 1000  # 最小面積閾值
    min_distance = 10  # 最小距離閾值

    # 繪製結果圖像
    result_image = np.zeros_like(image)
    for i in range(1, len(stats)):
        if stats[i][4] >= min_area:
            x, y = int(centroids[i][0]), int(centroids[i][1])
            cv2.circle(result_image, (x, y), 5, (255, 255, 255), -1)
            for j in range(1, len(stats)):
                if i != j and stats[j][4] >= min_area:
                    distance = np.sqrt((centroids[i][0] - centroids[j][0])**2 + (centroids[i][1] - centroids[j][1])**2)
                    if distance < min_distance:
                        cv2.line(result_image, (x, y), (int(centroids[j][0]), int(centroids[j][1])), (255, 255, 255), 2)

    return result_image

def generate_hat_filtered(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 調整核的大小和形狀

    # 應用Hat濾波器
    hat_filtered = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


class ImageDataLoader(Dataset):
    def __init__(self,  data_path, txt_file, transform=None):
        self.txt_file = txt_file
        self.txt_list = load_file_list(self.txt_file)
        self.data_path = data_path
        

        self.transform = transform

    
    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, index):



        img_path =  self.data_path + 'train/' + self.txt_list[index]
        mask_path = self.data_path + 'mask/' + self.txt_list[index]



        image = np.array(Image.open(img_path).convert('L'))




        mask = np.array(Image.open(mask_path).convert('L'))
        mask = mask / 255

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)

        # if np.random.rand() < 0.5:
        #     image2, mask2, _ = self.__getitem__(np.random.randint(0, len(self.txt_list)))
        #     # print(image2.shape)
        #     # image = cv2.resize(image, (64, 64))
        #     # mask = cv2.resize(mask, (64, 64))
        #     image = torch.from_numpy(image)
        #     mask = torch.from_numpy(mask)
        #     lam = np.random.beta(a=1.0, b=1.0)
        #     image = image.unsqueeze(0)

        #     image = torch.cat([lam*image, (1-lam)*image2], dim=0)
        #     mask = torch.cat([lam*mask, (1-lam)*mask2], dim=0)

        #     # print(image.shape)
            
        image = augmentations["image"] 
        mask = augmentations["mask"]
        # print('image',image.shape)
        # print('mask',mask.shape)
        return image, mask, img_path


class FakeDataLoader(Dataset):
    def __init__(self,  data_path, txt_file, transform=None):
        self.txt_file = txt_file
        self.txt_list = load_file_list(self.txt_file)
        self.data_path = data_path
        

        self.transform = transform

    
    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, index):



        img_path =  self.data_path + 'FakeAll/' + self.txt_list[index]



        image = np.array(Image.open(img_path).convert('L'))
        # plt.imshow(image)
        # plt.show()


        padding_image = np.zeros((224, 224))
        padding_image[48:176, 48:176] = image

        if self.transform:
            augmentations = self.transform(image=padding_image)

        # if np.random.rand() < 0.5:
        #     image2, mask2, _ = self.__getitem__(np.random.randint(0, len(self.txt_list)))
        #     # print(image2.shape)
        #     # image = cv2.resize(image, (64, 64))
        #     # mask = cv2.resize(mask, (64, 64))
        #     image = torch.from_numpy(image)
        #     mask = torch.from_numpy(mask)
        #     lam = np.random.beta(a=1.0, b=1.0)
        #     image = image.unsqueeze(0)

        #     image = torch.cat([lam*image, (1-lam)*image2], dim=0)
        #     mask = torch.cat([lam*mask, (1-lam)*mask2], dim=0)

        #     # print(image.shape)
        mask = torch.zeros((224, 224))
        image = augmentations["image"] 
        # print(image.shape)
        # print(mask.shape)
        # mask = augmentations["mask"]
        # print('image',image.shape)
        # print('mask',mask.shape)
        return image, mask, img_path

