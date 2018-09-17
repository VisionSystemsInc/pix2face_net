""" pix2face demo:
    loads an image, estimates PNCC + offset images, and outputs a 3D point cloud
"""
import pix2face
import numpy as np
from PIL import Image
import os.path

#----- parameters -----
img_fname = './data/CASIA_0000107_004.jpg'
output_dir = '.'
#----------------------

# Load the network weights
model = pix2face.test.load_pretrained_model()

# load the test image
img = np.array(Image.open(img_fname))

# run the image through te network
outputs = pix2face.test.test(model, [img,])
pncc = outputs[0][0]
offsets = outputs[0][1]

# PNCC + offsets = 3D point cloud
# The ply file can be loaded with a 3D viewer such as meshlab
basename = os.path.splitext(os.path.basename(img_fname))[0] 
output_point_cloud_fname = os.path.join(output_dir, basename + '.ply')
print('Writing ' + output_point_cloud_fname)
pix2face.data.save_ply(img, pncc, offsets, output_point_cloud_fname)
