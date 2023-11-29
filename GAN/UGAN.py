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
train_txt = '/home/student/Desktop/SegmentationThesis/data/train.txt'

train_transform = get_transform()

LR = 0.001
BATCH_SIZE = 32
EPOCHS = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = ImageDataLoader(
                            data_path=data_path,
                            txt_file = train_txt,
                            transform=train_transform
                              )

train_loader = DL(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

netG = UNet(n_channels = 1, n_classes=1).to(device)
netD = DiscriminatorU(in_channels=1).to(device)

criterion = nn.BCELoss()

L1loss = nn.L1Loss()

beta1 = 0.5
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.001, betas=(beta1, 0.999))

real_label = 1.
fake_label = 0.

img_list = []
mask_list = []
G_losses = []
D_losses = []
iter = 0

fix_idx = 0

 

for epoch in range(EPOCHS):
    for i, (img, mask, _) in enumerate(tqdm(train_loader)):
        img = img.float()
        img = (img + 1) / 2
        # print(img.shape)
        
        # imgnp = img[fix_idx].transpose(0, 2).transpose(0, 1).cpu().detach().numpy()
        # plt.imshow(imgnp)
        # plt.show()
        # 创建一个三通道的图像
        # rgb_image = torch.zeros((img.shape[0], 3, img.shape[2], img.shape[3]))

        # # 将原始图像的单通道值分别赋值给 RGB 图像的每个通道
        # for channel in range(3):
        #     rgb_image[:, channel, :, :] = img[:, 0, :, :]   # 使用不同的缩放因子

        # # 将范围调整为 [0, 1]
        # rgb_image = torch.clamp(rgb_image, 0, 1)

        # # print(rgb_image.shape)
        # # 将范围调整为 [0, 1]
        # # rgb_image = torch.clamp(rgb_image, -1, 1)

        # img = rgb_image
        # imgnp = img[fix_idx].transpose(0, 2).transpose(0, 1).cpu().detach().numpy()
        # plt.imshow(imgnp)
        # plt.show()
        img = img.to(device).float()
        mask = mask.to(device).float()
        mask = mask.unsqueeze(1)

        
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        netD.zero_grad()
        # Train with real
        # real_data = torch.cat((img, mask), 1)
        
        
        # print(real_data.shape)
        output = netD(img)
        
        # print(real_data.shape)
        out_shape = netD(img).shape
        label = torch.full(out_shape, real_label, device=device)
        errD_real = criterion(output, label)
        errD_real.backward()

        # Train with fake
        # mask1 = mask.clone()
        fake_image = netG(mask)
        # print(fake_image.shape)
        # fake_data = torch.cat((fake_image.detach(), mask), 1)  # Detach to avoid backpropagating through G
        label.fill_(fake_label)
        # print(fake_data.shape)
        output = netD(fake_image)
        errD_fake = criterion(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # output = netD(torch.cat((fake_image, mask), 1))
        fake_image = netG(mask)
        output = netD(fake_image)
        errG_adv = criterion(output, label)
        errG_L1 = L1loss(fake_image, img)
        errG = errG_adv + errG_L1
        errG.backward()
        optimizerG.step()

        # Print statistics
        if epoch % 10 == 0 and i == 0 :
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, EPOCHS, i, len(train_loader),
                     errD.item(), errG.item()))

            masknp = mask[fix_idx].transpose(0, 2).transpose(0, 1).cpu().detach().numpy()
            fakenp = fake_image[fix_idx].transpose(0, 2).transpose(0, 1).cpu().detach().numpy()


print(masknp.shape)
print(fakenp.shape)
plt.subplot(1, 3, 1)
plt.imshow(masknp)
plt.subplot(1, 3, 2)
plt.imshow(fakenp)
plt.subplot(1, 3, 3)
plt.imshow(img[fix_idx].transpose(0, 2).transpose(0, 1).cpu().detach().numpy())
plt.show()

# plt.figure(figsize=(15,15))
# plt.axis("off")
# plt.title("Fake Images")
# plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True),(1,2,0)))
# plt.show()
# plt.figure(figsize=(15,15))
# plt.axis("off")
# plt.title("Fake Images")
# plt.imshow(np.transpose(vutils.make_grid(mask, padding=2, normalize=True),(1,2,0)))
# plt.show()

torch.save(netG.state_dict(), 'UnetG.pth')
torch.save(netD.state_dict(), 'UnetD.pth')