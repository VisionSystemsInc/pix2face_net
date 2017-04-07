from __future__ import print_function
import skimage.io
import skimage.external.tifffile as tifffile
import torch
from torch.autograd import Variable

import data


def test():
    model = torch.load('/proj/janus/pix2face/unet.pth')
    input = skimage.io.imread('/proj/janus/data/CASIA-WebFace/CASIA-WebFace/0000045/001.jpg')
    input = data.prepare_input(input)
    mb = data.images_to_minibatch([input,])
    out = model.forward(Variable(mb))

    out = data.minibatch_to_images(out.data)
    tifffile.imsave('/tmp/output.tiff', out[0])

if __name__ == '__main__':
    test()
