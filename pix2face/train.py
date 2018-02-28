from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from . import network
from . import data


def train(input_dir, PNCC_dir, offsets_dir, face_box_dir,
          val_input_dir, val_PNCC_dir, val_offsets_dir, val_face_box_dir,
          output_dir,
          start_epoch=0):
    """ Train the Pix2Face Model
    """
    # intervals are in units of epochs
    minibatch_size = 4
    print_interval = 400 / minibatch_size
    save_interval = 8000 / minibatch_size
    validate_interval = 80000 / minibatch_size
    num_epochs = 60
    cuda = True

    log_filename = os.path.join(output_dir, 'train_loss.npy')
    val_loss_filename = os.path.join(output_dir, 'validation_loss.npy')
    log_epoch_filename = os.path.join(output_dir, 'train_loss_epoch_%d.npy')

    if offsets_dir is None:
        model = network.Pix2PNCCNet()
        model_filename = os.path.join(output_dir, 'pix2PNCC_unet.pth')
    else:
        model = network.Pix2FaceNet()
        model_filename = os.path.join(output_dir, 'pix2face_unet.pth')
        model_epoch_filename = os.path.join(output_dir, 'pix2face_unet_epoch_%d.pth')


    if start_epoch > 0:
        continue_model_filename = model_epoch_filename % (start_epoch - 1)
        model_state_dict = torch.load(continue_model_filename)
        model.load_state_dict(model_state_dict)

    if cuda:
        model.cuda()
    lr_max = 0.001
    optimizer = optim.Adam(model.parameters(), lr_max, betas=(0.5, 0.9))
    #optimizer = optim.SGD(model.parameters(), lr_max)

    train_set = data.Pix2FaceTrainingData(input_dir, PNCC_dir, offsets_dir, face_box_dir)
    val_set = data.Pix2FaceTrainingData(val_input_dir, val_PNCC_dir, val_offsets_dir, val_face_box_dir, jitter=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=minibatch_size, shuffle=True, num_workers=8, pin_memory=True)

    #loss_fn = nn.SmoothL1Loss()
    loss_fn = nn.L1Loss()
    if cuda:
        loss_fn = loss_fn.cuda()

    images_per_epoch = 100000
    num_minibatches_per_epoch = np.min((images_per_epoch/minibatch_size, len(train_loader)))
    num_minibatches = num_epochs*num_minibatches_per_epoch
    mb_loss = np.zeros(num_minibatches) + np.nan
    val_loss = np.zeros(num_epochs+1) + np.nan
    mb_loss_epoch = np.linspace(0,num_epochs, len(mb_loss))
    np.save(os.path.join(output_dir, 'train_loss_epoch_num.npy'), mb_loss_epoch)

    if start_epoch > 0:
        continue_log_filename = log_epoch_filename % (start_epoch - 1)
        prev_log = np.load(continue_log_filename)
        mb_loss[:len(prev_log)] = prev_log
        prev_val_loss = np.load(val_loss_filename)
        val_loss[:len(prev_val_loss)] = prev_val_loss

    np.save(val_loss_filename, val_loss)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        lr_decay_scale = 5.0 / num_epochs
        lr = lr_max / (2**(lr_decay_scale*epoch))
        print('Setting learning rate to ' + str(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
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
                    epoch, batch_idx * minibatch_size, minibatch_size*num_minibatches_per_epoch,
                    100. * batch_idx / num_minibatches_per_epoch, loss.data[0]))
            if batch_idx % save_interval == 0:
                np.save(log_filename, mb_loss)
                torch.save(model.state_dict(), model_filename)
                print('wrote ' + log_filename + ' and ' + model_filename)
            if batch_idx == (num_minibatches_per_epoch-1):
                print('stopping epoch.')
                break

        # Epoch Complete: Perform Validation after each epoch
        model.eval()
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, num_workers=8)
        val_mb_losses = np.zeros(len(val_loader)) + np.nan
        for v_batch_idx, (v_input, v_target) in enumerate(val_loader):
            if v_batch_idx % print_interval == 0:
                print('Validation: Minibatch {}/{}'.format(v_batch_idx, len(val_loader)))
            if cuda:
                v_input, v_target = v_input.cuda(), v_target.cuda()
            v_input, v_target = Variable(v_input), Variable(v_target)
            v_output = model(v_input)
            v_loss = loss_fn(v_output, v_target)
            val_mb_losses[v_batch_idx] = v_loss.data[0]
        val_loss[epoch+1] = np.mean(val_mb_losses)
        np.save(val_loss_filename, val_loss)
        print('Validation Loss = ' + str(val_loss[epoch+1]))

        # save (at a minimum) after every completed epoch
        np.save(log_epoch_filename % epoch, mb_loss)
        torch.save(model.state_dict(), model_epoch_filename % epoch)
        print('Epoch complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Face network training')
    parser.add_argument('--input_dir', required=True, help='directory containing input images')
    parser.add_argument('--PNCC_dir', required=True, help='directory containing target PNCC images')
    parser.add_argument('--offsets_dir', default=None, help='directory containing target offset images')
    parser.add_argument('--face_box_dir', default=None, help='directory containing image face bounding boxes, one file per image of form <left top right bottom>')
    parser.add_argument('--val_input_dir', required=True, help='directory containing validation input images')
    parser.add_argument('--val_PNCC_dir', required=True, help='directory containing validation target PNCC images')
    parser.add_argument('--val_offsets_dir', default=None, help='directory containing validation target offset images')
    parser.add_argument('--val_face_box_dir', default=None, help='directory containing image face bounding boxes, one file per image of form <left top right bottom>')
    parser.add_argument('--output_dir', required=True, help='directory to write model and logs to')
    parser.add_argument('--start_epoch', required=False, type=int, default=0)
    args = parser.parse_args()
    print('pid = ' + str(os.getpid()))
    train(args.input_dir, args.PNCC_dir, args.offsets_dir, args.face_box_dir,
          args.val_input_dir, args.val_PNCC_dir, args.val_offsets_dir, args.val_face_box_dir,
          args.output_dir,
          args.start_epoch)
