from __future__ import print_function
import os
import skimage.io
import skimage.external.tifffile as tifffile
import torch
from torch.autograd import Variable

import data


def test(model_filename, input_fname, output_dir):
    minibatch_size = 4

    model = torch.load(model_filename)
    if os.path.isdir(input_fname):
        input_filenames = [os.path.join(input_fname, f) for f in os.listdir(input_fname)]
    else:
        input_filenames = [input_fname,]
    output_filenames = [os.path.join(output_dir, os.path.splitext(os.path.basename(f))[0] + '.tiff') for f in input_filenames]

    num_inputs = len(input_filenames)
    for i_begin in range(0, num_inputs, minibatch_size):
        i_end = min(i+minibatch_size, num_inputs)
        minibatch_inputs = list()
        for i in range(i_begin, i_end):
            # load image
            input = skimage.io.imread(input_filenames[i])
            # normalize / convert to float
            minibatch_inputs.append( data.prepare_input(input) )

        # create minibatch
        mb = data.images_to_minibatch(minibatch_inputs)
        # run minibatch through the network
        out = model.forward(Variable(mb))
        # convert minibatch output to list of images
        minibatch_outputs = data.minibatch_to_images(out.data)
        # sanity check on size of output
        if len(minibatch_outputs) != len(minibatch_inputs):
            print('len(inputs) = ' + str(len(minibatch_inputs)))
            print('len(outputs) = ' + str(len(minibatch_outputs)))
            raise Exception('size of minibatch inputs and outputs do not match')
        # write out output images
        for i in range(i_begin, i_end):
            tifffile.imsave(output_filenames[i], minibatch_outputs[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Face network testing')
    parser.add_argument('--model', required=True, help='filename of trained model')
    parser.add_argument('--in', required=True, help='filename or directory of input image(s)')
    parser.add_argument('--output_dir', required=True, help='directory to write output to')
    args = parser.parse_args()
    test(args.model, args.in, args.out)
