from __future__ import print_function
import os
import signal
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import network
import data

def start_pdb(sig, frame):
    """ drop into the PDB debugger when signal is received.
    """
    import pdb
    pdb.Pdb().set_trace(frame)


def train(input_dir, PNCC_dir, offsets_dir, output_dir,
          start_epoch=0, continue_model_filename=None, continue_log_filename=None):
    """ Train the Pix2Face Model
    """
    print_interval = 100
    save_interval = 1000
    minibatch_size = 16
    num_epochs = 3
    cuda = True

    log_filename = os.path.join(output_dir, 'train_loss.npy')
    log_epoch_filename = os.path.join(output_dir, 'train_loss_epoch_%d.npy')

    if offsets_dir is None:
        model = network.Pix2PNCCNet()
        model_filename = os.path.join(output_dir, 'pix2PNCC_unet.pth')
    else:
        model = network.Pix2FaceNet()
        model_filename = os.path.join(output_dir, 'pix2face_unet.pth')
        model_epoch_filename = os.path.join(output_dir, 'pix2face_unet_epoch_%d.pth')

    if continue_model_filename is not None:
        model_state_dict = torch.load(continue_model_filename)
        model.load_state_dict(model_state_dict)

    if cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.95))

    train_set = data.Pix2FaceTrainingData(input_dir, PNCC_dir, offsets_dir)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=minibatch_size, shuffle=True, num_workers=8, pin_memory=True)

    loss_fn = nn.SmoothL1Loss()
    #loss_fn = nn.L1Loss()
    if cuda:
        loss_fn = loss_fn.cuda()

    num_minibatches_per_epoch = len(train_loader)
    num_minibatches = num_epochs*num_minibatches_per_epoch
    mb_loss = np.zeros(num_minibatches) + np.nan

    if continue_log_filename is not None:
        prev_log = np.load(continue_log_filename)
        mb_loss[:len(prev_log)] = prev_log

    for epoch in range(start_epoch, num_epochs):
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
                    np.save(log_filename, mb_loss)
                    torch.save(model.state_dict(), model_filename)
                    print('wrote ' + log_filename + ' and ' + model_filename)

        # save (at a minimum) after every completed epoch
        np.save(log_epoch_filename % epoch, mb_loss)
        torch.save(model.state_dict(), model_epoch_filename % epoch)
        print('Epoch complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Face network training')
    parser.add_argument('--input_dir', required=True, help='directory containing input images')
    parser.add_argument('--PNCC_dir', required=True, help='directory containing target PNCC images')
    parser.add_argument('--offsets_dir', default=None, help='directory containing target offset images')
    parser.add_argument('--output_dir', required=True, help='directory to write model and logs to')
    parser.add_argument('--start_epoch', required=False, type=int, default=0)
    parser.add_argument('--continue_model', required=False, default=None)
    parser.add_argument('--continue_log', required=False, default=None)
    args = parser.parse_args()
    signal.signal(signal.SIGUSR1, start_pdb)
    print('pid = ' + str(os.getpid()))
    train(args.input_dir, args.PNCC_dir, args.offsets_dir, args.output_dir,
          args.start_epoch, args.continue_model, args.continue_log)
