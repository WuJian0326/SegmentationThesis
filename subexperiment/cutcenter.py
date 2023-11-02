import os
import cv2
import numpy as np

txt_file = '/home/student/Desktop/SegmentationThesis/data/all.txt'
data_path = '/home/student/Desktop/SegmentationThesis/data/microglia/train/'

file_list = []
with open(txt_file, 'r') as f:
    for line in f:
        file_list.append(line.strip())

save_path = '/home/student/Desktop/SegmentationThesis/data/microglia/traincut128/'
os.makedirs(save_path, exist_ok=True)


for file in file_list:
    img = cv2.imread(data_path + file)
    imgcut = img[64:192, 64:192]
    cv2.imwrite(save_path + file, imgcut)


 