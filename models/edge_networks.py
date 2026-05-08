import torch
import torch.nn as nn


class DoubleConv(nn.Module):


    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class EdgeUNet(nn.Module):


    def __init__(self, n_channels=1, n_classes=1):
        super(EdgeUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes


        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))


        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)


        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)

        x = torch.cat([x2, x], dim=1)


        x = self.conv1(torch.cat([x, self.crop(x3, x)], dim=1))

        x = self.up2(x3)

        return self.sigmoid(self.outc(x_final))



class SimpleEdgeUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.e2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.e3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)


        self.b = DoubleConv(256, 512)


        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d1 = DoubleConv(512, 256)  # 256 from up + 256 from e3

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d2 = DoubleConv(256, 128)  # 128 + 128

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d3 = DoubleConv(128, 64)  # 64 + 64

        self.out = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        c1 = self.e1(x)
        p1 = self.pool1(c1)
        c2 = self.e2(p1)
        p2 = self.pool2(c2)
        c3 = self.e3(p2)
        p3 = self.pool3(c3)


        bn = self.b(p3)


        u1 = self.up1(bn)

        merge1 = torch.cat([u1, c3], dim=1)
        dc1 = self.d1(merge1)

        u2 = self.up2(dc1)
        merge2 = torch.cat([u2, c2], dim=1)
        dc2 = self.d2(merge2)

        u3 = self.up3(dc2)
        merge3 = torch.cat([u3, c1], dim=1)
        dc3 = self.d3(merge3)

        return self.sigmoid(self.out(dc3))