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

data_path = '/home/student/Desktop/SegmentationThesis/data/microglia/'
train_txt = '/home/student/Desktop/SegmentationThesis/data/train.txt'
train_transform = get_transform()
# 超參數

LEARNING_RATE = 0.0002


batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100



# Beta1 hyperparam for Adam optimizers
beta1 = 0.5


# 設備配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 資料集讀取

train_data = ImageDataLoader(
                            data_path=data_path,
                            txt_file = train_txt,
                            transform=train_transform
                              )
train_loader = DL(dataset=train_data, batch_size=batch_size, shuffle=True)

netG = Generator64().to(device)
netD = Discriminator64().to(device)
print(netG)
netG.apply(weights_init)
netD.apply(weights_init)
# netG.load_state_dict(torch.load('netGnormallize.pth'))
# netD.load_state_dict(torch.load('netDnormallize.pth'))


# 損失函數
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(beta1, 0.999))
real_label = 1.
fake_label = 0.
# 訓練
num_epochs = 150
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (data,_,_) in enumerate(train_loader, 0):
        data = data.float()

        # print(data.shape)
        crop_x = (224 - 64) // 2
        crop_y = (224 - 64) // 2

        # 使用切片操作裁剪 data
        data = data[:, :, crop_y:crop_y+64, crop_x:crop_x+64]
        # print(data.shape)
        rgb_image = torch.zeros((data.shape[0], 3, data.shape[2], data.shape[3]))

# 将单通道数据复制到三通道中
        rgb_image[:, 0, :, :] = data[:, 0, :, :]
        rgb_image[:, 1, :, :] = data[:, 0, :, :]
        rgb_image[:, 2, :, :] = data[:, 0, :, :]
        data = rgb_image
        # print(data.shape)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data.to(device) # what does cpu mean?
        # print(real_cpu.shape)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # print(output.shape)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
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


torch.save(netG.state_dict(), 'netG.pth')
torch.save(netD.state_dict(), 'netD.pth')

generate_img = netG(fixed_noise).detach().cpu()
plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(generate_img, padding=2, normalize=True),(1,2,0)))
plt.show()



