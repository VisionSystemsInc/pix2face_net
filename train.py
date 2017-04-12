from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import network
import data


def train(input_dir, PNCC_dir, offsets_dir, output_dir):
    learning_rate = 0.01
    momentum = 0.5
    print_interval = 100
    save_interval = 1000
    minibatch_size = 16
    num_epochs = 1
    cuda = True

    log_filename = os.path.join(output_dir, 'train_loss.npy')

    if offsets_dir is None:
        model = network.Pix2PNCCNet()
        model_filename = os.path.join(output_dir, 'pix2PNCC_unet.pth')
    else:
        model = network.Pix2FaceNet()
        model_filename = os.path.join(output_dir, 'pix2face_unet.pth')
    if cuda:
        model.cuda()
    #optimizer = optim.SGD(model.parameters(), learning_rate, momentum)
    optimizer = optim.Adam(model.parameters())

    train_set = data.Pix2FaceTrainingData(input_dir, PNCC_dir, offsets_dir)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=minibatch_size, shuffle=True, num_workers=8)

    #loss_fn = nn.SmoothL1Loss()
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
                output = model(input)
                loss = loss_fn(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mb_loss[num_minibatches_per_epoch*epoch + batch_idx] = loss.data[0]
                if batch_idx % print_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0]))
                if batch_idx % save_interval == 0:
                    print('writing ' + log_filename + ' and ' + model_filename)
                    np.save(log_filename, mb_loss)
                    torch.save(model.state_dict(), model_filename)

        # save (at a minimum) after every completed epoch
        print('Epoch complete: writing ' + log_filename + ' and ' + model_filename)
        np.save(log_filename, mb_loss)
        torch.save(model.state_dict(), model_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Face network training')
    parser.add_argument('--input_dir', required=True, help='directory containing input images')
    parser.add_argument('--PNCC_dir', required=True, help='directory containing target PNCC images')
    parser.add_argument('--offsets_dir', default=None, help='directory containing target offset images')
    parser.add_argument('--output_dir', required=True, help='directory to write model and logs to')
    args = parser.parse_args()
    train(args.input_dir, args.PNCC_dir, args.offsets_dir, args.output_dir)
