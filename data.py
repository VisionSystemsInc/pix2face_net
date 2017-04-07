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
    img = img.astype(np.float)
    # convert to expected size
    img = skimage.transform.resize(img, (256,256))
    # transform pixels to range (-0.5, 0.5)
    img /= 255 - 0.5
    return img


class Pix2FaceTrainingData(Dataset):
    def __init__(self, image_dir, target_dir):
        self.in_filenames = list()
        self.target_filenames = list()

        self.in_filenames = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        self.target_filenames = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir)])
        if len(self.in_filenames) != len(self.target_filenames):
            raise Exception('Different numbers of input and target images')

    def __len__(self):
        return len(self.in_filenames)

    def __getitem__(self, idx):
        # load images
        img = skimage.io.imread(self.in_filenames[idx]).astype(np.float)
        target = tifffile.imread(self.target_filenames[idx])

        if img.shape[0:2] != target.shape[0:2]:
            raise Exception('Inconsistent input and target image sizes')

        # convert to expected size
        img = skimage.transform.resize(img, (256,256))
        # transform pixels to range (-0.5, 0.5)
        img /= 255 - 0.5

        # transform target to expected size
        target = skimage.transform.resize(target, (256,256))
        # remove NaNs from target
        target[np.isnan(target)] = 0.0

        img = torch.Tensor(np.moveaxis(img, 2, 0))  # make num_channels first dimension
        target = torch.Tensor(np.moveaxis(target, 2, 0))  # make num_channels first dimension
        return img, target


def save_tiff(img, filename):
    tifffile.imsave(filename, img)
