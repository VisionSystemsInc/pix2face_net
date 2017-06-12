from __future__ import print_function
import os
import numpy as np

import skimage.io
import skimage.external.tifffile as tifffile
import skimage.transform
import skimage.morphology

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


def prepare_input(img, targets=None, jitter=False, min_crop_ratio=0.75, max_crop_ratio=1.0, max_crop_shift_ratio=0.1, crop_center_x=0.5, crop_center_y=0.5):
    # ensure input is a color image
    if len(img.shape) < 3 or img.shape[2] < 3:
        img = skimage.color.gray2rgb(img)
    # strip alpha layer, if present
    img = img[:,:,0:3]
    # convert to floating point
    img = img.astype(np.float)
    if targets is not None:
        num_targets = len(targets)
        for t in range(num_targets):
            targets[t] = targets[t].astype(np.float)

    # randomly resize and crop image
    mindim = np.min(img.shape[0:2])
    if jitter:
        # crop will be size 256x256
        # randomly resize such that crop has size ratio in desired range
        mindim_min = int(256/max_crop_ratio)
        mindim_max = int(256/min_crop_ratio)
        new_mindim = np.random.randint(mindim_min, mindim_max)
    else:
        new_mindim = 256
    scale = np.float(new_mindim)/mindim
    new_shape = np.ceil(scale*np.array(img.shape[0:2])).astype(np.int)

    img = skimage.transform.resize(img, new_shape, mode='constant')
    if targets is not None:
        for t in range(num_targets):
            targets[t] = skimage.transform.resize(targets[t], new_shape, mode='constant')

    # crop square
    if jitter:
        center_x = img.shape[1] * crop_center_x
        center_y = img.shape[0] * crop_center_y
        max_x = int(center_x + max_crop_shift_ratio*img.shape[1] - 128)
        max_x = min(max_x, img.shape[1] - 256)
        min_x = int(center_x - max_crop_shift_ratio*img.shape[1] - 128)
        min_x = max(min_x, 0)
        max_y = int(center_y + max_crop_shift_ratio*img.shape[0] - 128)
        max_y = min(max_y, img.shape[0] - 256)
        min_y = int(center_y - max_crop_shift_ratio*img.shape[0] - 128)
        min_y = max(min_y, 0)

        if max_x > min_x:
            xoff = np.random.randint(min_x,max_x)
        else:
            xoff = 0
        if max_y > min_y:
            yoff = np.random.randint(min_y,max_y)
        else:
            yoff = 0
    else:
        xoff = (img.shape[1] - 256)/2
        yoff = (img.shape[0] - 256)/2
    img = img[yoff:yoff+256,xoff:xoff+256,:]
    if targets is not None:
        for t in range(num_targets):
            targets[t] = targets[t][yoff:yoff+256, xoff:xoff+256,:]

    # jitter color values
    if jitter:
        gamma_mag = 0.4
        color_mag = 0.2
        # pick a random gamma correction factor
        gamma = np.max((0.0, 1 + gamma_mag * (np.random.random()-0.5)))
        # pick a random color balancing scheme
        color_scale = 1 - np.random.random(3) * color_mag
        color_scale /= np.max(color_scale)

        for c in range(3):
            img[:,:,c] = (np.power(img[:,:,c]*color_scale[c], gamma))

        img[img > 255] = 255
        img[img < 0] = 0

    # transform pixels to range (-0.5, 0.5)
    img = img/255 - 0.5
    # transform targets to range(-1,1)
    if targets is not None:
        for t in range(num_targets):
            targets[t] = targets[t]/255 * 2 - 1.0
        return img, targets
    else:
        return img


def prepare_output(img, input_shape, use_3DMM_bbox=True):
    # separate if two images concatenated
    if img.shape[2] == 6:
        imgs = [img[:,:,0:3].astype(np.float32), img[:,:,3:6].astype(np.float32)]
    elif img.shape[2] == 3:
        imgs = [img.astype(np.float32),]
    else:
        raise Exception('Unexpected image shape: ' + str(img.shape))

    # unnormalize pncc and offset values
    imgs[0] = unnormalize_PNCC(imgs[0], use_3DMM_bbox)
    if len(imgs) > 0:
        imgs[1] = unnormalize_offsets(imgs[1], use_3DMM_bbox)

    # convert to expected size (square aspect ratio)
    mindim = np.min(input_shape[0:2])
    imgs = [skimage.transform.resize(img, (mindim,mindim), mode='constant').astype(np.float32) for img in imgs]

    # create outputs of correct size
    imgs_out = [np.zeros((input_shape[0],input_shape[1],3), img.dtype) for img in imgs]
    # fill in center region with square crops
    if input_shape[0] > input_shape[1]:
        s = (input_shape[0] - input_shape[1])/2
        for i in range(len(imgs_out)):
            imgs_out[i][s:s+mindim,:,:] = imgs[i]
    else:
        s = (input_shape[1] - input_shape[0])/2
        for i in range(len(imgs_out)):
            imgs_out[i][:,s:s+mindim,:] = imgs[i]


    return imgs_out


def unnormalize_PNCC(pncc_in, use_3DMM_bbox=True):
    """ convert PNCC image with values in range (-1,1) to actual 3-d coordinates.
        Note that the bounding box values below must match those in face3d/semantic_map.cxx
    """
    if use_3DMM_bbox:
        min_val = np.array((-100.0, -130.0, -25.0))
        max_val = np.array((100.0, 100.0, 150.0))
    else:
        min_val = np.array((-100.0, -130.0, -120.0))
        max_val = np.array((100.0, 130.0, 120.0))
    scale = (max_val - min_val)/2.0
    offset = (max_val + min_val)/2.0
    pncc = pncc_in * scale + offset
    return pncc


def unnormalize_offsets(offsets_in, use_3DMM_bbox=True):
    """ convert offset image with values in range (-1,1) to actual 3-d coordinates.
        Note that the bounding box values below must match those in face3d/semantic_map.cxx
    """
    if use_3DMM_bbox:
        min_val = np.array((-25.0, -25.0, -25.0))
        max_val = np.array((25.0, 25.0, 25.0))
    else:
        min_val = np.array((-20.0, -20.0, -20.0))
        max_val = np.array((20.0, 20.0, 20.0))
    scale = (max_val - min_val)/2.0
    offset = (max_val + min_val)/2.0
    offsets = offsets_in * scale + offset
    return offsets

def normalize_PNCC(pncc_in, use_3DMM_bbox=True):
    """ convert pncc image with 3-d coordinates to values in range (-1,1)
        Note that the bounding box values below must match those in face3d/semantic_map.cxx
    """
    if use_3DMM_bbox:
        min_val = np.array((-100.0, -130.0, -25.0))
        max_val = np.array((100.0, 100.0, 150.0))
    else:
        min_val = np.array((-100.0, -130.0, -120.0))
        max_val = np.array((100.0, 130.0, 120.0))
    scale = (max_val - min_val)/2.0
    offset = (max_val + min_val)/2.0
    pncc = (pncc_in - offset ) / scale
    pncc[pncc < -1] = -1
    pncc[pncc > 1] = 1
    return pncc


def normalize_offsets(offsets_in, use_3DMM_bbox=True):
    """ convert offset image with real 3-d values to range (-1,1)
        Note that the bounding box values below must match those in face3d/semantic_map.cxx
    """
    if use_3DMM_bbox:
        min_val = np.array((-25.0, -25.0, -25.0))
        max_val = np.array((25.0, 25.0, 25.0))
    else:
        min_val = np.array((-20.0, -20.0, -20.0))
        max_val = np.array((20.0, 20.0, 20.0))
    scale = (max_val - min_val)/2.0
    offset = (max_val + min_val)/2.0
    offsets = (offsets_in - offset) / scale
    offsets[offsets < -1] = -1
    offsets[offsets > 1] = 1
    return offsets


def save_ply(img, pncc, offsets, filename):
    """ Save a point cloud with r,g,b taken from the input image
        pncc and offsets should be in unnormalized form
    """
    # create mask of valid points
    mag_sqrd_thresh = 100.0  # no valid points near origin
    mask = np.sum(pncc*pncc,axis=2) > mag_sqrd_thresh
    # erode mask to remove noisy points on the border
    mask = skimage.morphology.binary_erosion(mask, selem=np.ones((3,3),dtype=np.uint8))
    # extract valid 3d points and their colors
    pts = (pncc[mask,:] + offsets[mask,:]).transpose()
    colors = img[mask,:].transpose()
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
    def __init__(self, input_dir, target_PNCC_dir, target_offsets_dir=None, jitter=True):
        self.jitter = jitter
        self.min_crop_ratio = 0.6
        self.max_crop_ratio = 0.9
        self.max_crop_shift_ratio = 0.05
        self.crop_center_x = 0.5
        self.crop_center_y = 0.6
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

        target_PNCC = skimage.io.imread(self.target_PNCC_filenames[idx])

        if img.shape[0:2] != target_PNCC.shape[0:2]:
            print('img.shape = ' + str(img.shape))
            print('target_PNCC.shape = ' + str(target_PNCC.shape))
            raise Exception('Inconsistent input and target PNCC image sizes')

        targets = [target_PNCC,]

        if self.target_offsets_filenames is not None:
            target_offsets = skimage.io.imread(self.target_offsets_filenames[idx])
            if img.shape[0:2] != target_offsets.shape[0:2]:
                print('img.shape = ' + str(img.shape))
                print('target_offsets.shape = ' + str(target_offsets.shape))
                raise Exception('Inconsistent input and target offsets image sizes')
            targets.append(target_offsets)

            # transform offsets to expected size
            target_offsets = skimage.transform.resize(target_offsets, (256,256), mode='constant')

        img, targets = prepare_input(img, targets, jitter=self.jitter,
                                     min_crop_ratio=self.min_crop_ratio,
                                     max_crop_ratio=self.max_crop_ratio,
                                     max_crop_shift_ratio=self.max_crop_shift_ratio,
                                     crop_center_x=self.crop_center_x,
                                     crop_center_y=self.crop_center_y)
        img = torch.Tensor(np.moveaxis(img, 2, 0))  # make num_channels first dimension

        # concatenate target images together
        target_all = targets[0]
        for target in targets[1:]:
            target_all = np.concatenate((target_all, target),axis=2)

        target = torch.Tensor(np.moveaxis(target_all, 2, 0))  # make num_channels first dimension
        return img, target


def save_tiff(img, filename):
    tifffile.imsave(filename, img)
