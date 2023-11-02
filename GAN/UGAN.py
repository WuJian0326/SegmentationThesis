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
BATCH_SIZE = 64
EPOCHS = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = ImageDataLoader(
                            data_path=data_path,
                            txt_file = train_txt,
                            transform=train_transform
                              )
train_loader = DL(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

netG = UNet(n_channels = 1, n_classes=2).to(device)
netD = DiscriminatorU().to(device)

criterion = nn.BCELoss()

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
    for i,(img, mask, _) in enumerate(tqdm(train_loader)):
        img = img.to(device).float()
        mask = mask.to(device).float()
        mask = mask.unsqueeze(1)
        # noise = torch.randn(img.size(0), 1, img.size(2), img.size(3)).to(device).float()
        stack = torch.cat((img, mask), dim=1)
        if fix_idx == 0:
            stack_img = img
            fix_idx += 1
        real_img = torch.cat((img, mask), dim=1)
        fake_img = netG(img)
        label = torch.full((img.size(0),), real_label, dtype=torch.float, device=device)

        output = netD(real_img).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        fake = netD(fake_img.detach()).view(-1)
        label.fill_(fake_label)
        errD_fake = criterion(fake, label)
        errD_fake.backward()
        D_G_z1 = fake.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        fake = netD(fake_img).view(-1)


        # errGseg = criterion(fake_img[:, 1:2, :, :], mask)
        errG = criterion(fake, label)

        errG = errG 
        errG.backward()
        D_G_z2 = fake.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, EPOCHS, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        

        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iter % 100 == 0) or ((epoch == EPOCHS-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = netG(stack_img).detach().cpu()
                
                img = fake[:, 0:1, :, :]  # 第一个通道
                # mask = fake[:, 1:2, :, :]  # 第二个通道

            img_list.append(vutils.make_grid(img, padding=2, normalize=True))
            # mask_list.append(vutils.make_grid(mask, padding=2, normalize=True))
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

        iter += 1 

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True),(1,2,0)))
plt.show()
# plt.figure(figsize=(15,15))
# plt.axis("off")
# plt.title("Fake Images")
# plt.imshow(np.transpose(vutils.make_grid(mask, padding=2, normalize=True),(1,2,0)))
# plt.show()

torch.save(netG.state_dict(), 'UnetG.pth')
torch.save(netD.state_dict(), 'UnetD.pth')