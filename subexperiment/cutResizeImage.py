import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

data_path = "/home/student/Desktop/SegmentationThesis/data/microglia/train/"
mask_path = "/home/student/Desktop/SegmentationThesis/data/microglia/mask/"
all_txt = "/home/student/Desktop/SegmentationThesis/data/all.txt"

saving_path = "/home/student/Desktop/SegmentationThesis/data/microglia/mask128/"
if not os.path.exists(saving_path):
    os.makedirs(saving_path)


file_list = []
with open(all_txt, 'r') as file:
    for line in file:
        line = line.strip()  # 移除開頭和結尾的空格或換行符號

        file_list.append(line)

for i in range(len(file_list)):
    image = cv2.imread(data_path + file_list[i],0)
    mask = cv2.imread(mask_path + file_list[i],0)

    box = np.argwhere(mask)
    mini_x = min(box[:,0])
    mini_y = min(box[:,1])
    maxi_x = max(box[:,0])
    maxi_y = max(box[:,1])

    ROI = mask[mini_x:maxi_x,mini_y:maxi_y]
    ROI = cv2.resize(ROI,(128,128))
    save_mask = np.zeros_like(mask)
    save_mask[48:176,48:176] = ROI
    cv2.imwrite(saving_path + file_list[i],save_mask)

    # plt.subplot(1,2,1)
    # plt.imshow(save_mask)
    # plt.subplot(1,2,2)
    # plt.imshow(mask[mini_x:maxi_x,mini_y:maxi_y])
    # plt.show()


    # break

