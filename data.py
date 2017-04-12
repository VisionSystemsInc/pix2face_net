from __future__ import print_function
import os
import numpy as np

import skimage.io
import skimage.external.tifffile as tifffile
import skimage.transform

import torch
from torch.utils.data import Dataset


def images_to_minibatch(images):
    num_images = len(images)
    img_shape = images[0].shape
    mb_np = np.zeros((num_images, 3, img_shape[0], img_shape[1]))
    for i in range(num_images):
        img = np.moveaxis(images[i],2,0)
        mb_np[i,:,:,:] = img
    return torch.Tensor(mb_np)


def minibatch_to_images(mb):
    mb_np = np.moveaxis(mb.numpy(), 1, 3)
    images = list()
    for i in range(mb_np.shape[0]):
        images.append(mb_np[i,:,:,:].squeeze())
    return images


def prepare_input(img):
    # ensure input is a color image
    if len(img.shape) < 3 or img.shape[2] < 3:
        img = skimage.color.gray2rgb(img)
    # strip alpha layer, if present
    img = img[:,:,0:3]
    # convert to floating point
    img = img.astype(np.float)
    # convert to expected size
    img = skimage.transform.resize(img, (256,256))
    # transform pixels to range (-0.5, 0.5)
    img /= 255 - 0.5
    return img

def prepare_output(img, input_shape):
    # separate if two images concatenated
    if img.shape[2] == 6:
        imgs = (img[:,:,0:3], img[:,:,3:6])
    elif img.shape[2] == 3:
        imgs = (img,)
    else:
        raise Exception('Unexpected image shape: ' + str(img.shape))

    # convert to expected size
    imgs_out = [skimage.transform.resize(img, input_shape[0:2]) for img in imgs]
    return imgs_out


class Pix2FaceTrainingData(Dataset):
    def __init__(self, input_dir, target_PNCC_dir, target_offsets_dir=None):
        print('input_dir = ' + input_dir)
        print('target_PNCC_dir = ' + target_PNCC_dir)
        if target_offsets_dir is not None:
            print('target_offsets_dir = ' + target_offsets_dir)

        self.input_filenames = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir)])
        self.target_PNCC_filenames = sorted([os.path.join(target_PNCC_dir, fname) for fname in os.listdir(target_PNCC_dir)])
        if target_offsets_dir is None:
            self.target_offsets_filenames = None
        else:
            self.target_offsets_filenames = sorted([os.path.join(target_offsets_dir, fname) for fname in os.listdir(target_offsets_dir)])
        if len(self.input_filenames) != len(self.target_PNCC_filenames):
            raise Exception('Different numbers of input and target PNCC images')
        if self.target_offsets_filenames is not None and len(self.input_filenames) != len(self.target_offsets_filenames):
            raise Exception('Different numbers of input and target offsets images')
        print(str(len(self.input_filenames)) + ' total training images in dataset.')


    def __len__(self):
        return len(self.input_filenames)


    def __getitem__(self, idx):
        # load images
        img = skimage.io.imread(self.input_filenames[idx])
        target_ext = os.path.splitext(self.target_PNCC_filenames[idx])[1]

        target_PNCC = skimage.io.imread(self.target_PNCC_filenames[idx]).astype(np.float)
        target_PNCC /= 255 - 0.5

        if img.shape[0:2] != target_PNCC.shape[0:2]:
            print('img.shape = ' + str(img.shape))
            print('target_PNCC.shape = ' + str(target_PNCC.shape))
            raise Exception('Inconsistent input and target PNCC image sizes')

        # transform PNCC to expected size
        target_PNCC = skimage.transform.resize(target_PNCC, (256,256))

        if self.target_offsets_filenames is not None:
            target_offsets = skimage.io.imread(self.target_offsets_filenames[idx]).astype(np.float)
            target_offsets /= 255 - 0.5
            if img.shape[0:2] != target_offsets.shape[0:2]:
                print('img.shape = ' + str(img.shape))
                print('target_offsets.shape = ' + str(target_offsets.shape))
                raise Exception('Inconsistent input and target offsets image sizes')

            # transform offsets to expected size
            target_offsets = skimage.transform.resize(target_offsets, (256,256))

        img = prepare_input(img)
        img = torch.Tensor(np.moveaxis(img, 2, 0))  # make num_channels first dimension

        # concatenate target images together
        if self.target_offsets_filenames:
            target = np.concatenate((target_PNCC, target_offsets),axis=2)
        else:
            target = target_PNCC

        target = torch.Tensor(np.moveaxis(target, 2, 0))  # make num_channels first dimension
        return img, target


def save_tiff(img, filename):
    tifffile.imsave(filename, img)
