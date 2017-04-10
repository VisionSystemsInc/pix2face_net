from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from network import Unet
from data import Pix2FaceTrainingData


def train(input_dir, target_dir, output_dir):
    num_filters = 64
    channels_in = 3
    channels_out = 3
    learning_rate = 0.01
    momentum = 0.5
    print_interval = 100
    save_interval = 1000
    num_epochs = 1
    cuda = True

    log_filename = os.path.join(output_dir, 'train_loss.npy')
    model_filename = os.path.join(output_dir, 'unet.pth')

    model = Unet(num_filters, channels_in, channels_out)
    if cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum)

    train_set = Pix2FaceTrainingData(input_dir, target_dir)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)

    loss_fn = nn.L1Loss()
    if cuda:
        loss_fn = loss_fn.cuda()

    num_minibatches_per_epoch = len(train_loader)
    num_minibatches = num_epochs*num_minibatches_per_epoch
    mb_loss = np.zeros(num_minibatches) + np.nan

    for epoch in range(num_epochs):
        for batch_idx, (input, target) in enumerate(train_loader):
                if cuda:
                    input, target = input.cuda(), target.cuda()
                input, target = Variable(input), Variable(target)
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                mb_loss[num_minibatches_per_epoch*epoch + batch_idx] = loss.data[0]
                if batch_idx % print_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0]))
                if batch_idx % save_interval == 0:
                    np.save(log_filename, mb_loss)
                    torch.save(model, model_filename)

        # save (at a minimum) after every completed epoch
        np.save(log_filename, mb_loss)
        torch.save(model, model_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Face network training')
    parser.add_argument('--input_dir', required=True, help='directory containing input images')
    parser.add_argument('--target_dir', required=True, help='directory containing target images')
    parser.add_argument('--output_dir', required=True, help='directory to write model and logs to')
    args = parser.parse_args()
    train(args.input_dir, args.target_dir, args.output_dir)
