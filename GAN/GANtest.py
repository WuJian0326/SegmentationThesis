from GANmodel import *
import torch
import torchvision
from torchvision.utils import save_image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



nz = 100

netD = Discriminator().to(device)
netG = Generator(nz = nz).to(device)
netD.load_state_dict(torch.load('netD100.pth'))
netG.load_state_dict(torch.load('netG100.pth'))

# test



# 定义一些测试超参数
num_images_to_generate = 3000  # 生成图像的数量
output_dir = f'./FakeImage/GAN{nz}'  # 保存生成图像的目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 生成并保存图像
generated_images = []
num = 0
while True:
    noise = torch.randn(1, nz, 1, 1).to(device)
    with torch.no_grad():
        generated_image = netG(noise)
    with torch.no_grad():
        is_real = netD(generated_image)
        # print(is_real)

    if is_real.mean() > 0.6:
        generated_images.append(generated_image)
        num += 1
        print(num)
    if num >= num_images_to_generate:
        break
# for i in range(num_images_to_generate):
#     # 生成随机噪声
#     noise = torch.randn(1, 100, 1, 1).to(device)

#     # 使用生成器生成图像
#     with torch.no_grad():  # 不计算梯度
#         generated_image = netG(noise)
    
#     # 判别器评估生成的图像
#     with torch.no_grad():
#         is_real = netD(generated_image)
    
#     # 如果判别器将图像视为真实图像，则保存
#     if is_real.mean() > 0.5:
#         generated_images.append(generated_image)

# 保存生成的图像
for i, image in enumerate(generated_images):
    image_filename = os.path.join(output_dir, f'generated_image{nz}_{i}.png')
    save_image(image, image_filename)

print(f'{len(generated_images)} 生成的图像已保存到 {output_dir} 目录中。')
