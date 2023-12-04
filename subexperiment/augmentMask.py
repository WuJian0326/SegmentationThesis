import cv2
import os
import random
from matplotlib import pyplot as plt
import numpy as np


mask_path = '/home/student/Desktop/SegmentationThesis/data/microglia/mask/'
mask2_path = '/home/student/Desktop/SegmentationThesis/data/microglia/mask128/'
img_path = '/home/student/Desktop/SegmentationThesis/data/microglia/train/'

augment_mask_path = '/home/student/Desktop/SegmentationThesis/data/microglia/augment_mask/'

if not os.path.exists(augment_mask_path):
    os.makedirs(augment_mask_path)

mask_list = os.listdir(mask_path)


for mask in mask_list:
    mask1 = cv2.imread(mask_path + mask)
    mask2 = cv2.imread(mask2_path + mask)
    ## origin mask
    cv2.imwrite(augment_mask_path + mask.split('.')[0] + '_origin.png', mask1)
    ###random vertical flip
    mask1_vertical = cv2.flip(mask1, 0)
    mask2_vertical = cv2.flip(mask2, 0)
    # plt.subplot(1,2,1)
    # plt.title("vertical")
    # plt.imshow(mask1_vertical)

    # plt.subplot(1,2,2)
    # plt.imshow(mask2_vertical)
    # plt.show()

    cv2.imwrite(augment_mask_path + mask.split('.')[0] + '_vertical.png', mask1_vertical)

    ###random horizontal flip
    mask1_horizontal = cv2.flip(mask1, 1)
    mask2_horizontal = cv2.flip(mask2, 1)
    # plt.subplot(1,2,1)
    # plt.title("horizontal")
    # plt.imshow(mask1_horizontal)

    # plt.subplot(1,2,2)
    # plt.imshow(mask2_horizontal)
    # plt.show()
    cv2.imwrite(augment_mask_path + mask.split('.')[0] + '_horizontal.png', mask1_horizontal)

    ###random rotate
    angle = random.uniform(0, 180) 
    M = cv2.getRotationMatrix2D((mask1.shape[0]/2,mask1.shape[1]/2),angle,1)
    mask1_rotate = cv2.warpAffine(mask1,M,(mask1.shape[0],mask1.shape[1]))
    mask2_rotate = cv2.warpAffine(mask2,M,(mask2.shape[0],mask2.shape[1]))
    # plt.subplot(1,2,1)
    # plt.title("rotate")
    # plt.imshow(mask1_rotate, cmap='gray', )

    # plt.subplot(1,2,2)
    # plt.imshow(mask2_rotate)
    # plt.show()
    cv2.imwrite(augment_mask_path + mask.split('.')[0] + '_rotate.png', mask1_rotate)

    ### random rotate2
    angle = random.uniform(180, 360) 
    M = cv2.getRotationMatrix2D((mask1.shape[0]/2,mask1.shape[1]/2),angle,1) 
    mask1_rotate2 = cv2.warpAffine(mask1,M,(mask1.shape[0],mask1.shape[1]))
    mask2_rotate2 = cv2.warpAffine(mask2,M,(mask2.shape[0],mask2.shape[1]))
    # plt.subplot(1,2,1)
    # plt.title("rotate2")
    # plt.imshow(mask1_rotate2, cmap='gray', )

    # plt.subplot(1,2,2)
    # plt.imshow(mask2_rotate2)
    # plt.show()
    cv2.imwrite(augment_mask_path + mask.split('.')[0] + '_rotate2.png', mask1_rotate2)

    ###close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    mask1_close = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    mask2_close = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask1_close = np.where(mask1_close > 100, 255, 0)
    mask2_close = np.where(mask2_close > 100, 255, 0)

    # plt.subplot(1,2,1)
    # plt.title("close")
    # plt.imshow(mask1_close)

    # plt.subplot(1,2,2)
    # plt.imshow(mask2_close)
    # plt.show()
    cv2.imwrite(augment_mask_path + mask.split('.')[0] + '_close.png', mask1_close)

    # ###hat



