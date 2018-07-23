from __future__ import print_function
import os
import numpy as np

import skimage.io
import skimage.external.tifffile as tifffile
import skimage.transform
import skimage.morphology

import torch
from torch.utils.data import Dataset

from collections import namedtuple

Rect = namedtuple('Rect',('left','top','right','bottom'))

MIN_TRAIN_SCALE = 1.0
MAX_TRAIN_SCALE = 1.8


def shift_and_scale_rect(rect, center_x=0.5, center_y=0.5, scale=1.0, make_square=False):
    """
    Given a Rect, shift, scale, and optionally make square
    Assumes image coordinates, i.e. "top" means min y
    No bounds checking is performed.
    Returns the new Rect.
    """
    height = rect.bottom - rect.top
    width = rect.right - rect.left
    new_center_x = rect.left + center_x*width
    new_center_y = rect.top + center_y*height
    new_half_height = scale*height/2.0
    new_half_width = scale*width/2.0
    if make_square:
        new_half_width = new_half_height = min(new_half_width, new_half_height)
    new_top = int(new_center_y - new_half_height)
    new_bottom = int(new_center_y + new_half_height)
    new_left = int(new_center_x - new_half_width)
    new_right = int(new_center_x + new_half_width)
    return Rect(left=new_left, top=new_top, right=new_right, bottom=new_bottom)


def sanitize_crop(rect, img_shape):
    """ convert a crop region to image and crop bounds such that:
    crop[crop_rect.top:crop_rect.bottom,..] = img[img_rect.top:img_rect.bottom,..]
    """
    # clip bounds
    top = max(0, rect.top)
    bottom = min(img_shape[0], rect.bottom)
    left = max(0, rect.left)
    right = min(img_shape[1], rect.right)
    # compute offset at which to place image data
    crop_top = top - rect.top
    crop_bottom = crop_top + (bottom - top)
    crop_left = left - rect.left
    crop_right = crop_left + (right - left)
    crop_rect = Rect(crop_left, crop_top, crop_right, crop_bottom)
    img_rect = Rect(left, top, right, bottom)
    return img_rect, crop_rect


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


def prepare_input(img, targets=None, face_box=None, jitter=False, min_scale=MIN_TRAIN_SCALE, max_scale=MAX_TRAIN_SCALE, max_shift=0.1, crop_center_x=0.5, crop_center_y=0.5, rot_range=45.0, target_background_vals=None):
    # ensure input is a color image
    if len(img.shape) < 3 or img.shape[2] < 3:
        img = skimage.color.gray2rgb(img)
    # strip alpha layer, if present
    img = img[:,:,0:3]

    if targets is not None and target_background_vals is None:
        target_background_vals = [np.zeros(target.shape[2]) for target in targets]

    # convert to floating point
    img = img.astype(np.float)
    if targets is not None:
        num_targets = len(targets)
        for t in range(num_targets):
            targets[t] = targets[t].astype(np.float)
            # transform targets to range(-1,1)
            targets[t] = targets[t]/255 * 2 - 1.0

    # randomly resize and crop image
    if jitter:
        scale = min_scale + np.random.random_sample()*(max_scale-min_scale)
        center_x = crop_center_x + (2*np.random.random_sample()-1)*max_shift
        center_y = crop_center_y + (2*np.random.random_sample()-1)*max_shift
    else:
        if face_box is None:
            # use full input image, don't scale
            face_box = Rect(0,0,img.shape[1],img.shape[0])
            scale = 1
        else:
            # use provided bounding box, use mean training scale
            scale = (min_scale + max_scale)/2.0
        center_x = crop_center_x
        center_y = crop_center_y
    face_box = shift_and_scale_rect(face_box, center_x, center_y, scale, make_square=True)

    img_rect, crop_rect = sanitize_crop(face_box, img.shape)
    crop_width = face_box.right - face_box.left
    crop_height = face_box.bottom - face_box.top
    # create crops
    crop = np.zeros((crop_height, crop_width, img.shape[2]), img.dtype)
    crop[crop_rect.top:crop_rect.bottom, crop_rect.left:crop_rect.right] \
        = img[img_rect.top:img_rect.bottom, img_rect.left:img_rect.right]
    img = crop
    if targets is not None:
        for t in range(num_targets):
            crop = np.zeros((crop_height, crop_width, img.shape[2]), img.dtype)
            crop += target_background_vals[t]
            crop[crop_rect.top:crop_rect.bottom, crop_rect.left:crop_rect.right] \
                = targets[t][img_rect.top:img_rect.bottom, img_rect.left:img_rect.right]
            targets[t] = crop

    # randomly rotate image
    if jitter and rot_range > 0:
        rot_angle = np.random.uniform(-rot_range, rot_range)
        img = skimage.transform.rotate(img, rot_angle, mode='edge')
        if targets is not None:
            for t in range(num_targets):
                targets[t] = skimage.transform.rotate(targets[t], rot_angle, mode='edge')

    # resize to final shape for processing
    final_shape = (256,256)
    img = skimage.transform.resize(img, final_shape, mode='edge', anti_aliasing=False)
    if targets is not None:
        for t in range(num_targets):
            targets[t] = skimage.transform.resize(targets[t], final_shape, mode='edge', anti_aliasing=False)

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
    if targets is not None:
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
    imgs = [skimage.transform.resize(im, (mindim,mindim), mode='constant', anti_aliasing=True).astype(np.float32) for im in imgs]

    # create outputs of correct size
    imgs_out = [np.zeros((input_shape[0],input_shape[1],3), im.dtype) for im in imgs]
    # fill in center region with square crops
    if input_shape[0] > input_shape[1]:
        s = (input_shape[0] - input_shape[1])//2
        for i in range(len(imgs_out)):
            imgs_out[i][s:s+mindim,:,:] = imgs[i]
    else:
        s = (input_shape[1] - input_shape[0])//2
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
    def __init__(self, input_dir, target_PNCC_dir, target_offsets_dir=None, face_box_dir=None, jitter=True, use_3DMM_bbox=True):
        self.jitter = jitter
        self.min_scale = MIN_TRAIN_SCALE
        self.max_scale = MAX_TRAIN_SCALE
        self.max_shift = 0.25
        self.crop_center_x = 0.5
        self.crop_center_y = 0.5
        self.rot_range = 20.0
        self.target_background_vals = [normalize_PNCC((0.0,0.0,0.0), use_3DMM_bbox), normalize_offsets((0.0, 0.0, 0.0), use_3DMM_bbox)]
        print('input_dir = ' + input_dir)
        print('target_PNCC_dir = ' + target_PNCC_dir)
        if target_offsets_dir is not None:
            print('target_offsets_dir = ' + target_offsets_dir)
        if face_box_dir is not None:
            print('face_box_dir = ' + face_box_dir)

        self.input_filenames = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir)])
        self.target_PNCC_filenames = sorted([os.path.join(target_PNCC_dir, fname) for fname in os.listdir(target_PNCC_dir)])
        if target_offsets_dir is None:
            self.target_offsets_filenames = None
        else:
            self.target_offsets_filenames = sorted([os.path.join(target_offsets_dir, fname) for fname in os.listdir(target_offsets_dir)])
        if face_box_dir is None:
            self.face_box_filenames = None
        else:
            self.face_box_filenames = sorted([os.path.join(face_box_dir, fname) for fname in os.listdir(face_box_dir)])

        if len(self.input_filenames) != len(self.target_PNCC_filenames):
            raise Exception('Different numbers of input and target PNCC images')
        if self.target_offsets_filenames is not None and len(self.input_filenames) != len(self.target_offsets_filenames):
            raise Exception('Different numbers of input and target offsets images')
        if self.face_box_filenames is not None and len(self.input_filenames) != len(self.face_box_filenames):
            raise Exception('Different numbers of input and face box files')
        print(str(len(self.input_filenames)) + ' total training images in dataset.')

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        # load images
        img = skimage.io.imread(self.input_filenames[idx])

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

        # get face bounding box
        if self.face_box_filenames is None:
            face_box = Rect(left=0,top=0,right=img.shape[1],bottom=img.shape[0])
        else:
            with open(self.face_box_filenames[idx],'r') as fd:
                vals = [int(val) for val in fd.readline().split()]
            face_box = Rect(left=vals[0], top=vals[1], right=vals[2], bottom=vals[3])

        # preprocess imagery
        img, targets = prepare_input(img, targets, face_box=face_box,
                                     jitter=self.jitter,
                                     min_scale=self.min_scale,
                                     max_scale=self.max_scale,
                                     max_shift=self.max_shift,
                                     crop_center_x=self.crop_center_x,
                                     crop_center_y=self.crop_center_y,
                                     rot_range=self.rot_range,
                                     target_background_vals=self.target_background_vals)
        img = torch.Tensor(np.moveaxis(img, 2, 0))  # make num_channels first dimension

        # concatenate target images together
        target_all = targets[0]
        for target in targets[1:]:
            target_all = np.concatenate((target_all, target),axis=2)

        target = torch.Tensor(np.moveaxis(target_all, 2, 0))  # make num_channels first dimension
        return img, target


def save_tiff(img, filename):
    tifffile.imsave(filename, img)
