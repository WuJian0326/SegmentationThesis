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
import segmentation_models_pytorch as smp

from DataLoader import *
from GANmodel import *
from model.unet_model import UNet

data_path = '/home/student/Desktop/SegmentationThesis/data/microglia/'
train_txt = '/home/student/Desktop/SegmentationThesis/data/test.txt'

train_transform = get_transform()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Generator_model = UNet(n_channels = 1, n_classes=1).to(device)
Generator_model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7

        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset) 
        ).to(device)



# for block in netG.encoder.blocks:
    # block.conv2.dropout = nn.Dropout(p=0.1)
Generator_model.encoder.conv1.dropout = nn.Dropout(p=0.5)
for block in Generator_model.encoder.layer1:
    block.conv1.dropout = nn.Dropout(p=0.5)
    block.conv2.dropout = nn.Dropout(p=0.5)
for block in Generator_model.encoder.layer2:
    block.conv1.dropout = nn.Dropout(p=0.5)
    block.conv2.dropout = nn.Dropout(p=0.5)
for block in Generator_model.encoder.layer3:
    block.conv1.dropout = nn.Dropout(p=0.5)
    block.conv2.dropout = nn.Dropout(p=0.5)
for block in Generator_model.encoder.layer4:
    block.conv1.dropout = nn.Dropout(p=0.5)
    block.conv2.dropout = nn.Dropout(p=0.5)


Generator_model.load_state_dict(torch.load('resUnetG.pth'))

test_data = Pix2pixloader(
                            data_path=data_path,
                            # txt_file = train_txt,
                            transform=train_transform
                              )
test_loader = DL(dataset=test_data, batch_size=1, shuffle=True)

for i, (_, mask, mask_path) in enumerate(tqdm(test_loader)):
    mask = mask.to(device).float()
    mask = mask.unsqueeze(1)
    
    mask_path = mask_path[0].split('/')[-1]
    # print(mask_path)
    # img = img.to(device).float()
    masknp = mask[0].transpose(0,1).transpose(1,2).cpu().numpy()
    with torch.no_grad():
        # imgnp = img[0].transpose(0,1).transpose(1,2).cpu().numpy()
        # imgnp += 1
        # imgnp /= 2
        p1 = Generator_model(mask)
        p1 = torch.sigmoid(p1)
        # print(p1.shape)
        
        out = p1[0].transpose(0,1).transpose(1,2).cpu().numpy()
        out = out * 255
        out = out.astype(np.uint8)
        # plt.subplot(1,2,1)
        # plt.imshow(masknp)
        # plt.subplot(1,2,2)
        # plt.imshow(out)
        # plt.show()
        # print(out.shape)
        ## save gray image
        # break
        cv2.imwrite(f'/home/student/Desktop/SegmentationThesis/data/microglia/GAN_augment_image/{mask_path}', out, )






