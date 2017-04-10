from __future__ import print_function

import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, num_filters, channels_in=3, channels_out=3):
        super(Unet, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, num_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(num_filters * 8 * 2, num_filters * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(num_filters * 8 * 2, num_filters * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(num_filters * 8 * 2, num_filters * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(num_filters * 8 * 2, num_filters * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(num_filters * 4 * 2, num_filters * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(num_filters * 2 * 2, num_filters, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(num_filters * 2, channels_out, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(num_filters)
        self.batch_norm2 = nn.BatchNorm2d(num_filters * 2)
        self.batch_norm4 = nn.BatchNorm2d(num_filters * 4)
        self.batch_norm8 = nn.BatchNorm2d(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (num_filters) x 128 x 128
        e2 = self.batch_norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.batch_norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.batch_norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.batch_norm8(self.conv5(self.leaky_relu(e4)))
        # state size is (num_filters x 8) x 8 x 8
        e6 = self.batch_norm8(self.conv6(self.leaky_relu(e5)))
        # state size is (num_filters x 8) x 4 x 4
        e7 = self.batch_norm8(self.conv7(self.leaky_relu(e6)))
        # state size is (num_filters x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        d1_ = self.dropout(self.batch_norm8(self.dconv1(self.relu(e8))))
        # state size is (num_filters x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.dropout(self.batch_norm8(self.dconv2(self.relu(d1))))
        # state size is (num_filters x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.dropout(self.batch_norm8(self.dconv3(self.relu(d2))))
        # state size is (num_filters x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.batch_norm8(self.dconv4(self.relu(d3)))
        # state size is (num_filters x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.batch_norm4(self.dconv5(self.relu(d4)))
        # state size is (num_filters x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.batch_norm2(self.dconv6(self.relu(d5)))
        # state size is (num_filters x 2) x 64 x 64
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.batch_norm(self.dconv7(self.relu(d6)))
        # state size is (num_filters) x 128 x 128
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.relu(d7))
        # state size is (nc) x 256 x 256
        output = self.tanh(d8)
        return output


class Pix2PNCCNet(Unet):
    def __init__(self):
        Unet.__init__(self, num_filters=64, channels_in=3, channels_out=3)
