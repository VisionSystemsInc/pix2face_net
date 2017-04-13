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
    # crop to square aspect ratio
    if img.shape[0] > img.shape[1]:
        mindim = img.shape[1]
        s = (img.shape[0] - img.shape[1])/2
        img = img[s:s+mindim,:,:]
    elif img.shape[1] > img.shape[0]:
        mindim = img.shape[0]
        s = (img.shape[1] - img.shape[0])/2
        img = img[:,s:s+mindim,:]
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

    # convert to expected size (square aspect ratio)
    mindim = np.min(input_shape[0:2])
    imgs = [skimage.transform.resize(img, (mindim,mindim)) for img in imgs]

    # create outputs of correct size
    imgs_out = [np.zeros((input_shape[0],input_shape[1],3), img.dtype) for img in imgs]
    print('mindim = ' + str(mindim))
    # fill in center region with square crops
    if input_shape[0] > input_shape[1]:
        s = (input_shape[0] - input_shape[1])/2
        print('s = ' + str(s))
        for i in range(len(imgs_out)):
            imgs_out[i][s:s+mindim,:,:] = imgs[i]
    else:
        s = (input_shape[1] - input_shape[0])/2
        for i in range(len(imgs_out)):
            imgs_out[i][:,s:s+mindim,:] = imgs[i]


    return imgs_out


def unnormalize_PNCC(pncc_in):
    """ convert PNCC image with values in range (-1,1) to actual 3-d coordinates.
        Note that the bounding box values below must match those in face3d/semantic_map.cxx
    """
    min_val = np.array((-100.0, -130.0, -120.0))
    max_val = np.array((100.0, 130.0, 120.0))
    scale = (max_val - min_val)/2.0
    offset = (max_val + min_val)/2.0
    pncc = pncc_in * scale + offset
    return pncc


def unnormalize_offsets(offsets_in):
    """ convert offset image with values in range (-1,1) to actual 3-d coordinates.
        Note that the bounding box values below must match those in face3d/semantic_map.cxx
    """
    min_val = np.array((-20.0, -20.0, -20.0))
    max_val = np.array((20.0, 20.0, 20.0))
    scale = (max_val - min_val)/2.0
    offset = (max_val + min_val)/2.0
    offsets = offsets_in * scale + offset
    return offsets


def save_ply(img, pncc, offsets, filename):
    """ Save a point cloud with r,g,b taken from the input image
        pncc and offsets should be in normalized form (values in range (-1,1))
    """
    # extract valid 3d points and their colors
    img3d = unnormalize_PNCC(pncc) + unnormalize_offsets(offsets)
    pts_all = np.moveaxis(img3d,2,0).reshape(3,-1)
    mag_thresh = 10.0  # no valid points near origin
    mask = np.sum(pts_all*pts_all,axis=0) > mag_thresh
    pts = pts_all[:,mask]
    colors = np.moveaxis(img,2,0).reshape(3,-1)[:,mask]
    # write ply file
    with open(filename,'w') as fd:
        fd.write('ply\n')
        fd.write('format ascii 1.0\n')
        fd.write('comment Created by pix2face.\n')
        fd.write('element vertex ' + str(pts.shape[1]) + '\n')
        fd.write('property float x\n')
        fd.write('property float y\n')
        fd.write('property float z\n')
        fd.write('property uint8 red\n')
        fd.write('property uint8 green\n')
        fd.write('property uint8 blue\n')
        fd.write('end_header\n')
        for i in range(pts.shape[1]):
            fd.write('%0.3f %0.3f %0.3f %d %d %d\n' % (pts[0,i],pts[1,i],pts[2,i],colors[0,i],colors[1,i],colors[2,i]))


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

        target_PNCC = skimage.io.imread(self.target_PNCC_filenames[idx]).astype(np.float)/255.0
        target_PNCC = target_PNCC  * 2 - 1.0  # range -1,1

        if img.shape[0:2] != target_PNCC.shape[0:2]:
            print('img.shape = ' + str(img.shape))
            print('target_PNCC.shape = ' + str(target_PNCC.shape))
            raise Exception('Inconsistent input and target PNCC image sizes')

        # transform PNCC to expected size
        target_PNCC = skimage.transform.resize(target_PNCC, (256,256))

        if self.target_offsets_filenames is not None:
            target_offsets = skimage.io.imread(self.target_offsets_filenames[idx]).astype(np.float)/255.0
            target_offsets = target_offsets * 2 - 1.0  # range -1,1
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
