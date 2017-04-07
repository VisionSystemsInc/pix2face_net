from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from network import Unet
from data import Pix2FaceTrainingData


def train():
    num_filters = 32
    channels_in = 3
    channels_out = 3
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 1
    num_epochs = 100

    img_dir = '/proj/janus/data/pix2face_data/in'
    target_dir = '/proj/janus/data/pix2face_data/target'

    model = Unet(num_filters, channels_in, channels_out)
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum)

    train_set = Pix2FaceTrainingData(img_dir, target_dir)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)

    loss_fn = nn.L1Loss()

    for epoch in range(num_epochs):
        for batch_idx, (input, target) in enumerate(train_loader):
                input, target = Variable(input), Variable(target)
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0]))

    torch.save(model, '/proj/janus/pix2face/unet.pth')

if __name__ == '__main__':
    train()
