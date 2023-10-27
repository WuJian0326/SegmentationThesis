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
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001


batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Learning rate for optimizers
lr = 0.001

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
train_loader = DL(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

netG = Generator().to(device)
netD = Discriminator().to(device)
netE = Encoder().to(device)

netE.apply(weights_init)
netG.apply(weights_init)
netD.apply(weights_init)


# 損失函數
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=LEARNING_RATE, betas=(beta1, 0.999))

real_label = 1.
fake_label = 0.
# 訓練
num_epochs = 100
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
        crop_x = (224 - 128) // 2
        crop_y = (224 - 128) // 2

        # 使用切片操作裁剪 data
        data = data[:, :, crop_y:crop_y+128, crop_x:crop_x+128]
        # print(data.shape)
#         rgb_image = torch.zeros((data.shape[0], 3, data.shape[2], data.shape[3]))

# # 将单通道数据复制到三通道中
#         rgb_image[:, 0, :, :] = data[:, 0, :, :]
#         rgb_image[:, 1, :, :] = data[:, 0, :, :]
#         rgb_image[:, 2, :, :] = data[:, 0, :, :]
        
        print(data.shape)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        
        

        netD.zero_grad()
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # 經過 Encoder 的真實圖像
        encoded_real = netE(real_cpu).to(device)
        encoded_real = encoded_real.view(batch_size, -1)

        linear = nn.Linear(100, 1*128*128).to(device)
        encoded_real = linear(encoded_real).to(device)
        encoded_real = encoded_real.view(batch_size, 1, 128, 128)
        # print('E(x)',encoded_real.shape)
        # print('x',real_cpu.shape)
        # 串聯 x 和 E(x)
        combined_input = torch.cat([real_cpu, encoded_real], dim=1)



        output_real = netD(combined_input).view(-1)
        errD_real = criterion(output_real, label)
        errD_real.backward()
        D_x = output_real.mean().item()

        # 生成假圖像
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)

        # 經過 Encoder 的假圖像
        z = noise.view(batch_size, -1)
        # print("z",z.shape)
        # print("Gz",fake.shape)
        linear = nn.Linear(100, 1*128*128).to(device)
        resizeNoise = linear(z).to(device)
        resizeNoise = resizeNoise.view(batch_size, 1, 128, 128)

        combined_GZ = torch.cat([fake, resizeNoise], dim=1)
        # print("combined",combined_GZ.shape)
        output_fake = netD(combined_GZ).view(-1)


        label.fill_(fake_label)
        errD_fake = criterion(output_fake, label)
        errD_fake.backward()
        D_G_z1 = output_fake.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        netE.zero_grad()
        encoded_x = netE(real_cpu)
        encoded_x = encoded_x.view(batch_size, -1)
        encoded_x = linear(encoded_x)
        encoded_x = encoded_x.view(batch_size, 1, 128, 128)
        output = netD(torch.cat([real_cpu, encoded_x], dim=1))  
        errE = criterion(output, torch.ones_like(output))
        errE.backward()
        optimizerE.step()
        ###########################
        netG.zero_grad()
        fake = netG(noise)
        noise = noise.view(batch_size, -1)
        noise = linear(noise)
        noise = noise.view(batch_size, 1, 128, 128)

        output = netD(torch.cat([fake, noise], dim=1)) 

        errG = criterion(output, torch.ones_like(output))
        errG.backward()
        D_G_z2 = output.mean().item()
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
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1

torch.save(netG.state_dict(), 'netG.pth')
torch.save(netD.state_dict(), 'netD.pth')

generate_img = netG(fixed_noise).detach().cpu()
plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(generate_img, padding=2, normalize=True),(1,2,0)))
plt.show()



