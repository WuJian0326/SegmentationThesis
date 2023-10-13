import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=224, nc=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        output = self.main(input)
        return output

# Discriminator 

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_filters=32):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(),
            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*8),
            nn.ReLU(),
            nn.Conv2d(num_filters*8, num_filters*16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_filters*16, num_filters*32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 使用平均池化将输出池化到 1x1
            nn.Flatten(),  # 将输出展平
            nn.Linear(num_filters*32, 1),  # 全连接层输出 1
            nn.Sigmoid()
        )

        
    def forward(self, x):
        return self.net(x)
