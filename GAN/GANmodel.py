import torch
import torch.nn as nn
import torch.nn.functional as F

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True)
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)
    
# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x):
#         # print(self.up(x).shape)

#         return self.conv(self.up(x))
    
# class Generator(nn.Module):
#     def __init__(self, nz = 100, ngf=64, nc=3):
#         super(Generator, self).__init__()
#         self.up1 = Up(nz, ngf * 16)
#         self.up2 = Up(ngf * 16, ngf * 8)
#         self.up3 = Up(ngf * 8, ngf * 4)
#         self.up4 = Up(ngf * 4, ngf * 2)
#         self.up5 = Up(ngf * 2, ngf)
#         self.up6 = Up(ngf, ngf // 2)
#         self.up7 = Up(ngf // 2, nc)
#         self.outc = nn.Tanh()

#     def forward(self, x):
#         # print(x.shape)
#         x1 = self.up1(x)
#         # print(x1.shape)
#         x2 = self.up2(x1)
#         # print(x2.shape)
#         x3 = self.up3(x2)
#         # print(x3.shape)
#         x4 = self.up4(x3)
#         # print(x4.shape)
#         x5 = self.up5(x4)
#         # print(x5.shape)
#         x6 = self.up6(x5)
#         # print(x6.shape)
#         output = self.up7(x6)
#         # print(output.shape)
#         output = self.outc(x6)
#         # print(output.shape)
#         return output

# Generator
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()

            # input is Z, going into a convolution
        self.c1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.b1 = nn.BatchNorm2d(ngf * 8)
        self.r1 = nn.ReLU(True)
            # state size. (ngf*8) x 4 x 4
        self.c2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.b2 = nn.BatchNorm2d(ngf * 4)
        self.r2 = nn.ReLU(True)
            # state size. (ngf*4) x 8 x 8
        self.c3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.b3 = nn.BatchNorm2d(ngf * 2)
        self.r3 = nn.ReLU(True)
            # state size. (ngf*2) x 16 x 16
        self.c4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.b4 = nn.BatchNorm2d(ngf)
        self.r4 = nn.ReLU(True)

        self.c5 = nn.ConvTranspose2d(ngf * 1, 32, 4, 2, 1, bias=False)
        self.b5 = nn.BatchNorm2d(32)
        self.r5 = nn.ReLU(True)
            # state size. (ngf) x 32 x 32
        self.c6 = nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=False)
        # self.c6 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(32, nc, 3, 1, 1, bias=False)
        # )
        self.t1 = nn.Tanh()
            # state size. (nc) x 64 x 64
        

    def forward(self, input):
        # print(input.shape)
        x = self.c1(input)
        x = self.b1(x)
        x = self.r1(x)
        # print(x.shape)
        x = self.c2(x)
        x = self.b2(x)
        x = self.r2(x)
        # print(x.shape)
        x = self.c3(x)
        x = self.b3(x)
        x = self.r3(x)
        # print(x.shape)
        x = self.c4(x)
        x = self.b4(x)
        x = self.r4(x)
        # print(x.shape)
        x = self.c5(x)
        x = self.b5(x)
        x = self.r5(x)
        # print(x.shape)
        x = self.c6(x)
        output = self.t1(x)
        # print(output.shape)
        return output
    

class Generator256(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator256, self).__init__()

            # input is Z, going into a convolution
        self.c1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.b1 = nn.BatchNorm2d(ngf * 8)
        self.r1 = nn.ReLU(True)
            # state size. (ngf*8) x 4 x 4
        self.c2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.b2 = nn.BatchNorm2d(ngf * 4)
        self.r2 = nn.ReLU(True)
            # state size. (ngf*4) x 8 x 8
        self.c3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.b3 = nn.BatchNorm2d(ngf * 2)
        self.r3 = nn.ReLU(True)
            # state size. (ngf*2) x 16 x 16
        self.c4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.b4 = nn.BatchNorm2d(ngf)
        self.r4 = nn.ReLU(True)

        self.c5 = nn.ConvTranspose2d(ngf * 1, 32, 4, 2, 1, bias=False)
        self.b5 = nn.BatchNorm2d(32)
        self.r5 = nn.ReLU(True)

        self.c6 = nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False)
        self.b6 = nn.BatchNorm2d(16)
        self.r6 = nn.ReLU(True)
            # state size. (ngf) x 32 x 32
        self.c7 = nn.ConvTranspose2d(16, nc, 4, 2, 1, bias=False)
        # self.c6 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(32, nc, 3, 1, 1, bias=False)
        # )
        self.t1 = nn.Tanh()
            # state size. (nc) x 64 x 64
        

    def forward(self, input):
        # print(input.shape)
        x = self.c1(input)
        x = self.b1(x)
        x = self.r1(x)
        # print(x.shape)
        x = self.c2(x)
        x = self.b2(x)
        x = self.r2(x)
        # print(x.shape)
        x = self.c3(x)
        x = self.b3(x)
        x = self.r3(x)
        # print(x.shape)
        x = self.c4(x)
        x = self.b4(x)
        x = self.r4(x)
        # print(x.shape)
        x = self.c5(x)
        x = self.b5(x)
        x = self.r5(x)
        # print(x.shape)
        x = self.c6(x)
        x = self.b6(x)
        x = self.r6(x)
        # print(x.shape)
        x = self.c7(x)
        output = self.t1(x)
        # print(output.shape)
        return output




class Generator64(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator64, self).__init__()

            # input is Z, going into a convolution
        self.c1 = nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False)
        self.b1 = nn.BatchNorm2d(ngf * 4)
        self.r1 = nn.ReLU(True)
            # state size. (ngf*8) x 4 x 4
        self.c2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.b2 = nn.BatchNorm2d(ngf * 2)
        self.r2 = nn.ReLU(True)
            # state size. (ngf*4) x 8 x 8
        self.c3 = nn.ConvTranspose2d(ngf * 2, ngf , 4, 2, 1, bias=False)
        self.b3 = nn.BatchNorm2d(ngf )
        self.r3 = nn.ReLU(True)


        self.c5 = nn.ConvTranspose2d(ngf * 1, 32, 4, 2, 1, bias=False)
        self.b5 = nn.BatchNorm2d(32)
        self.r5 = nn.ReLU(True)
            # state size. (ngf) x 32 x 32
        self.c6 = nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=False)
        # self.c6 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(32, nc, 3, 1, 1, bias=False)
        # )
        self.t1 = nn.Tanh()
            # state size. (nc) x 64 x 64
        

    def forward(self, input):
        # print(input.shape)
        x = self.c1(input)
        x = self.b1(x)
        x = self.r1(x)
        # print(x.shape)
        x = self.c2(x)
        x = self.b2(x)
        x = self.r2(x)
        # print(x.shape)
        x = self.c3(x)
        x = self.b3(x)
        x = self.r3(x)

        # print(x.shape)
        x = self.c5(x)
        x = self.b5(x)
        x = self.r5(x)
        # print(x.shape)
        x = self.c6(x)
        output = self.t1(x)
        # print(output.shape)
        return output
    
class Generator64(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator64, self).__init__()

            # input is Z, going into a convolution
        self.c1 = nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False)
        self.b1 = nn.BatchNorm2d(ngf * 4)
        self.r1 = nn.ReLU(True)
            # state size. (ngf*8) x 4 x 4
        self.c2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.b2 = nn.BatchNorm2d(ngf * 2)
        self.r2 = nn.ReLU(True)
            # state size. (ngf*4) x 8 x 8
        self.c3 = nn.ConvTranspose2d(ngf * 2, ngf , 4, 2, 1, bias=False)
        self.b3 = nn.BatchNorm2d(ngf )
        self.r3 = nn.ReLU(True)


        self.c5 = nn.ConvTranspose2d(ngf * 1, 32, 4, 2, 1, bias=False)
        self.b5 = nn.BatchNorm2d(32)
        self.r5 = nn.ReLU(True)
            # state size. (ngf) x 32 x 32
        self.c6 = nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=False)
        # self.c6 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(32, nc, 3, 1, 1, bias=False)
        # )
        self.t1 = nn.Tanh()
            # state size. (nc) x 64 x 64
        

    def forward(self, input):
        # print(input.shape)
        x = self.c1(input)
        x = self.b1(x)
        x = self.r1(x)
        # print(x.shape)
        x = self.c2(x)
        x = self.b2(x)
        x = self.r2(x)
        # print(x.shape)
        x = self.c3(x)
        x = self.b3(x)
        x = self.r3(x)

        # print(x.shape)
        x = self.c5(x)
        x = self.b5(x)
        x = self.r5(x)
        # print(x.shape)
        x = self.c6(x)
        output = self.t1(x)
        # print(output.shape)
        return output
    
# class Generator(nn.Module):
#     def __init__(self, nz=100, ngf=64, nc=3):
#         super(Generator, self).__init__()

#         # input is Z, going into a convolution
#         self.c1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
#         self.b1 = nn.BatchNorm2d(ngf * 8)
#         self.r1 = nn.ReLU(True)
#         # state size. (ngf*8) x 4 x 4

#         self.c2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.c2_conv = nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1, bias=False)
#         self.b2 = nn.BatchNorm2d(ngf * 4)
#         self.r2 = nn.ReLU(True)
#         # state size. (ngf*4) x 8 x 8

#         self.c3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.c3_conv = nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=False)
#         self.b3 = nn.BatchNorm2d(ngf * 2)
#         self.r3 = nn.ReLU(True)
#         # state size. (ngf*2) x 16 x 16

#         self.c4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.c4_conv = nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=False)
#         self.b4 = nn.BatchNorm2d(ngf)
#         self.r4 = nn.ReLU(True)

#         self.c5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.c5_conv = nn.Conv2d(ngf, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.b5 = nn.BatchNorm2d(32)
#         self.r5 = nn.ReLU(True)
#         # state size. (ngf) x 32 x 32

#         self.c6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.c6_conv = nn.Conv2d(32, nc, kernel_size=3, stride=1, padding=1, bias=False)
#         self.t1 = nn.Tanh()
#         # state size. (nc) x 64 x 64

#     def forward(self, input):
#         # print(input.shape)
#         x = self.c1(input)
#         # print(x.shape)
#         # x = self.c1_conv(x)
#         x = self.b1(x)
#         x = self.r1(x)
#         # print(x.shape)
#         x = self.c2(x)
#         x = self.c2_conv(x)
#         x = self.b2(x)
#         x = self.r2(x)
#         # print(x.shape)
#         x = self.c3(x)
#         x = self.c3_conv(x)
#         x = self.b3(x)
#         x = self.r3(x)
#         # print(x.shape)
#         x = self.c4(x)
#         x = self.c4_conv(x)
#         x = self.b4(x)
#         x = self.r4(x)
#         # print(x.shape)
#         x = self.c5(x)
#         x = self.c5_conv(x)
#         x = self.b5(x)
#         x = self.r5(x)
#         # print(x.shape)
#         x = self.c6(x)
#         x = self.c6_conv(x)
#         output = self.t1(x)
#         # print(output.shape)
#         return output

    
class Discriminator256(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
        super(Discriminator256, self).__init__()
        
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
            nn.Conv2d(8, 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=4, stride=1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)




# Discriminator 


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
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
    

class Discriminator64(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
        super(Discriminator64, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(),
            nn.Conv2d(num_filters*4, 8, kernel_size=4, stride=2, padding=1, bias=False),
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
    



class Discriminator(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
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
    

class DiscriminatorU(nn.Module):
    def __init__(self, in_channels=2, num_filters=64):
        super(DiscriminatorU, self).__init__()
        

        self.c1 = nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1, bias=False)
        self.c2 = nn.Conv2d(num_filters, num_filters, kernel_size=4, stride=2, padding=1, bias=False)
        self.r1 = nn.ReLU()
        self.c3 = nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.c4 = nn.Conv2d(num_filters*2, num_filters*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(num_filters*2)
        self.r2 = nn.ReLU()
        self.c5 = nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=0, bias=False)
        # self.c6 = nn.Conv2d(num_filters*4, num_filters*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(num_filters*4)
        self.r3 = nn.ReLU()
        self.c7 = nn.Conv2d(num_filters*4, 8, kernel_size=3, stride=1, padding=0, bias=False)
        self.b3 = nn.BatchNorm2d(8)
        self.r4 = nn.ReLU()
        self.c8 = nn.Conv2d(8, 1, kernel_size=4, stride=1, bias=True)
        self.outc = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.c1(x)
        # print(x.shape)
        x = self.c2(x)
        # print(x.shape)
        x = self.r1(x)
        x = self.c3(x)
        # print(x.shape)
        x = self.c4(x)
        # print(x.shape)
        x = self.b1(x)
        x = self.r2(x)
        x = self.c5(x)
        # print(x.shape)
        # x = self.c6(x)
        # print(x.shape)
        x = self.b2(x)
        x = self.r3(x)
        x = self.c7(x)
        # print(x.shape)
        x = self.b3(x)
        x = self.r4(x)
        x = self.c8(x)
        # print(x.shape)
        output = self.outc(x)

        return output
    


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)