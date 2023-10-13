from GANmodel import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


netD = Discriminator().to(device)
netG = Generator().to(device)
netD.load_state_dict(torch.load('netD.pth'))
netG.load_state_dict(torch.load('netG.pth'))

# test

import torch
import torchvision
from torchvision.utils import save_image
import os

# 定义一些测试超参数
num_images_to_generate = 10  # 生成图像的数量
output_dir = 'generated_images'  # 保存生成图像的目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 生成并保存图像
for i in range(num_images_to_generate):
    # 生成随机噪声
    noise = torch.randn(1, 100, 1, 1).to(device)

    # 使用生成器生成图像
    with torch.no_grad():  # 不计算梯度
        generated_image = netG(noise)

    # 保存生成的图像
    image_filename = os.path.join(output_dir, f'generated_image_{i}.png')
    save_image(generated_image, image_filename)

print(f'{num_images_to_generate} 生成的图像已保存到 {output_dir} 目录中。')