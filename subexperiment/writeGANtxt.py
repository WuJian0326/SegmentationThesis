import os
import random

# 資料資料夾路徑
folder_path = '/home/student/Desktop/SegmentationThesis/GAN/FakeImage/FakeAll'

file_names = os.listdir(folder_path)

def save_to_txt(file_names, file_path):
    with open(file_path, 'w') as f:
        for file in file_names:
            f.write(file + '\n')

save_to_txt(file_names, '/home/student/Desktop/SegmentationThesis/GAN/FakeImage/FakeAll.txt')