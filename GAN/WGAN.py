from GANmodel import Generator, weights_init
import torch
import torchvision
import sys
import torch.optim as optim
sys.path.append('..')
from torch.utils.data import DataLoader as DL
from torchvision.utils import save_image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from DataLoader import *
from WGANutils import *
import torchvision.utils as vutils

data_path = '/home/student/Desktop/SegmentationThesis/data/microglia/'
train_txt = '/home/student/Desktop/SegmentationThesis/data/all.txt'
train_transform = get_transform()
# 超參數

LEARNING_RATE = 0.0001


batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

num_epochs = 250


# 設備配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 資料集讀取

train_data = ImageDataLoader(
                            data_path=data_path,
                            txt_file = train_txt,
                            transform=train_transform
                              )
train_loader = DL(dataset=train_data, batch_size=batch_size, shuffle=True)



class Discriminator(torch.nn.Module):
    def __init__(self, channels = 3):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


netD = Discriminator().to(device)
netG = Generator().to(device)
# netD.load_state_dict(torch.load('WGANnetD.pth'))
# netG.load_state_dict(torch.load('WGANnetG.pth'))
netD.apply(weights_init)
netG.apply(weights_init)


lr = 1e-4
betas = (.9, .99)
oprimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas)
oprimizerD = optim.Adam(netD.parameters(), lr=lr, betas=betas)

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1.
fake_label = 0.
losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}

num_steps = 0
critic_iterations = 5
iters = 0
img_list = []

for epoch in range(num_epochs):
    # For each batch in the dataloader


    for i, (data,_,_) in enumerate(tqdm(train_loader), 0):
        num_steps += 1

        data = data.float()

        # print(data.shape)
        crop_x = (224 - 128) // 2
        crop_y = (224 - 128) // 2

        # 使用切片操作裁剪 data
        data = data[:, :, crop_y:crop_y+128, crop_x:crop_x+128]
        # print(data.shape)
        rgb_image = torch.zeros((data.shape[0], 3, data.shape[2], data.shape[3]))

# 将单通道数据复制到三通道中
        rgb_image[:, 0, :, :] = data[:, 0, :, :]
        rgb_image[:, 1, :, :] = data[:, 0, :, :]
        rgb_image[:, 2, :, :] = data[:, 0, :, :]
        data = rgb_image

        netD.zero_grad()

        real_cpu = data.to(device) 
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        genetate = netG(noise)

        d_real = netD(real_cpu)
        d_gerenate = netD(genetate)


        gradiant = gradient_penalty(real_cpu, genetate, netD)
        # losses['GP'].append(gradiant.data[0])
        
        oprimizerD.zero_grad()
        d_loss = d_gerenate.mean() - d_real.mean() + gradiant
        d_loss.backward()

        oprimizerD.step()

        # losses['D'].append(d_loss.data[0])
        if num_steps % critic_iterations == 0:
            oprimizerG.zero_grad()
            generated_data = netG(noise)
            d_generated = netD(generated_data)
            g_loss = - d_generated.mean()
            g_loss.backward()
            oprimizerG.step()

            # losses['G'].append(g_loss.data[0])


        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            # plt.figure(figsize=(15,15))
            # plt.axis("off")
            # plt.title("Fake Images")
            # plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)))
            # plt.show()
            
        iters += 1

torch.save(netG.state_dict(), 'WGANnetG.pth')
torch.save(netD.state_dict(), 'WGANnetD.pth')

generate_img = netG(fixed_noise).detach().cpu()
plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(generate_img, padding=2, normalize=True),(1,2,0)))
plt.show()
