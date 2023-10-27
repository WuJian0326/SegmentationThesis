import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

img = plt.imread('../data/microglia/high_quality/CR1S10_id1_R_R.png')
path = '../data/microglia/high_quality/'
path2 = '../data/microglia/train/'
path3 = '../data/microglia/mask/'


for filename in os.listdir(path):
    if filename.endswith('.png'):
        img = plt.imread(path + filename)
        img = np.array(img)
        img = img * 255
        img = img.astype(np.uint8)
        plt.imsave(path3 + filename, img, cmap='gray')



