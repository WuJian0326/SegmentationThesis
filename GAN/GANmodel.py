import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 1, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

# Discriminator 


class Discriminator(nn.Module):
    def __init__(self, in_channels=6, num_filters=64):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(),
            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*8),
            nn.ReLU(),
            nn.Conv2d(num_filters*8, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=4, stride=1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)
    

class Encoder(nn.Module):
    def __init__(self, in_channels=3, num_filters=64, latent_dim=100):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=5, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters*2, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(),
            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*8),
            nn.ReLU(),
            nn.Conv2d(num_filters*8, latent_dim, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU()

        )

        self.fc = nn.Linear(num_filters*8, latent_dim)

    def forward(self, x):
        x = self.net(x)

        return x
    

        

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)