from model.unet_model import UNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as DL
from torchvision import datasets
from torchvision.transforms import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from DataLoader import *
from GANmodel import *
from model.unet_model import UNet

data_path = '/home/student/Desktop/SegmentationThesis/data/microglia/'
train_txt = '/home/student/Desktop/SegmentationThesis/data/test.txt'

train_transform = get_transform()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Generator_model = UNet(n_channels = 1, n_classes=1).to(device)
Generator_model.load_state_dict(torch.load('UnetG.pth'))

test_data = ImageDataLoader(
                            data_path=data_path,
                            txt_file = train_txt,
                            transform=train_transform
                              )
test_loader = DL(dataset=test_data, batch_size=1, shuffle=True)

for i, (img, mask, _) in enumerate(tqdm(test_loader)):
    mask = mask.to(device).float()
    mask = mask.unsqueeze(1)

    img = img.to(device).float()

    with torch.no_grad():
        p1 = Generator_model(mask)
        # print(p1.shape)
        out = p1[0].transpose(0,1).transpose(1,2).cpu().numpy()
        plt.subplot(1,2,1)
        plt.imshow(mask[0].transpose(0,1).transpose(1,2).cpu().numpy(), cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(out, cmap='gray')
        plt.show()

