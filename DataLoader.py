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

        ToTensorV2(),
    ])

    return transform

def get_vaild_transform():
    transform = A.Compose([
        A.Resize(224, 224),

        ToTensorV2(),
    ])
    return transform


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
        image = np.expand_dims(image, axis=-1)
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = mask / 255

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)

            mask = augmentations["mask"]
            image = augmentations["image"]

        return image, mask, img_path


