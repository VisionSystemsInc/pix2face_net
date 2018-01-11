from __future__ import print_function
import os
import argparse
import numpy as np
import skimage.io
import skimage.external.tifffile as tifffile
import torch
from torch.autograd import Variable

import data
import network


def load_model(model_filename):
    """ load the pytorch model from disk """
    print('loading ' + str(model_filename) + ' ...')
    model = network.Pix2FaceNet()
    model_state_dict = torch.load(model_filename)
    model.load_state_dict(model_state_dict)
    print('...done.')
    return model


def test(model, inputs, face_boxes=None, cuda_device=None, use_3DMM_bbox=True):
    """ run the network on inputs, return list of numpy arrays """
    model.eval()
    minibatch_size = 8
    if cuda_device is not None:
        model = model.cuda(cuda_device)

    single_input = False
    if type(inputs) != list and type(inputs) != tuple:
        if type(inputs) == np.ndarray:
            single_input = True
            inputs = [inputs,]
            if face_boxes is not None:
                face_boxes = [face_boxes,]
        else:
            raise Exception('Unexpected input type ' + str(type(inputs)))

    num_inputs = len(inputs)
    outputs = list()
    for i_begin in range(0, num_inputs, minibatch_size):
        i_end = min(i_begin+minibatch_size, num_inputs)
        minibatch_inputs = list()
        input_shapes = list()
        for i in range(i_begin, i_end):
            # save original image shapes for later
            input_shapes.append(inputs[i].shape)
            # normalize / convert to float
            if face_boxes is None:
                face_box = None
            else:
                face_box = face_boxes[i]
            minibatch_inputs.append( data.prepare_input(inputs[i], face_box=face_box) )

        # create minibatch
        mb = data.images_to_minibatch(minibatch_inputs)
        if cuda_device is not None:
            mb = mb.cuda(cuda_device)
        # run minibatch through the network
        out = model(Variable(mb))
        if cuda_device is not None:
            out = out.data.cpu()
        else:
            out = out.data
        # convert minibatch output to list of images
        minibatch_outputs = data.minibatch_to_images(out)
        # sanity check on size of output
        if len(minibatch_outputs) != len(minibatch_inputs):
            print('len(inputs) = ' + str(len(minibatch_inputs)))
            print('len(outputs) = ' + str(len(minibatch_outputs)))
            raise Exception('size of minibatch inputs and outputs do not match')
        # extract output images from tensor
        for mb_i in range(len(minibatch_inputs)):
            i = i_begin + mb_i
            imgs_out = data.prepare_output(minibatch_outputs[mb_i], input_shapes[mb_i], use_3DMM_bbox)
            outputs.append(imgs_out)

    if single_input:
        return outputs[0]
    return outputs


def test_files(model_filename, input, output_dir, cuda_device=None, output_format='tiff', use_3DMM_bbox=True):
    """ run the network stored in model_filename on image(s) listed in input, write results to disk """
    chunk_size = 32  # maximum number of images to load at once
    model = load_model(model_filename)

    if type(input) == list:
        input_filenames = input
    elif os.path.isdir(input):
        input_filenames = [os.path.join(input, f) for f in os.listdir(input)]
    else:
        input_filenames = [input,]
    output_PNCC_filenames = [os.path.join(output_dir, os.path.splitext(os.path.basename(f))[0] + '_PNCC.' + output_format) for f in input_filenames]
    output_offsets_filenames = [os.path.join(output_dir, os.path.splitext(os.path.basename(f))[0] + '_offsets.' + output_format) for f in input_filenames]

    num_inputs = len(input_filenames)
    for i_begin in range(0, num_inputs, chunk_size):
        i_end = min(i_begin+chunk_size, num_inputs)
        chunk_inputs = list()
        for i in range(i_begin, i_end):
            # load image
            input = skimage.io.imread(input_filenames[i])
            chunk_inputs.append( input )
        chunk_outputs = test(model, chunk_inputs, cuda_device, use_3DMM_bbox)
        if len(chunk_outputs) != len(chunk_inputs):
            print('len(inputs) = ' + str(len(chunk_inputs)))
            print('len(outputs) = ' + str(len(chunk_outputs)))
            raise Exception('size of chunk inputs and outputs do not match')
        # write out output images
        for chunk_i in range(len(chunk_inputs)):
            i = i_begin + chunk_i
            if output_format == 'tiff':
                tifffile.imsave(output_PNCC_filenames[i], chunk_outputs[chunk_i][0])
                tifffile.imsave(output_offsets_filenames[i], chunk_outputs[chunk_i][1])
            else:
                pncc_out = data.normalize_PNCC(chunk_outputs[chunk_i][0], use_3DMM_bbox)
                pncc_out = (pncc_out/2.0 * 255).astype(np.uint8)
                skimage.io.imsave(output_PNCC_filenames[i], pncc_out)
                offsets_out = data.normalize_offsets(chunk_outputs[chunk_i][1], use_3DMM_bbox)
                offsets_out = (offsets_out/2.0 * 255).astype(np.uint8)
                skimage.io.imsave(output_offsets_filenames[i], offsets_out)
    return zip(output_PNCC_filenames, output_offsets_filenames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Face network testing')
    parser.add_argument('--model', required=True, help='filename of trained model')
    parser.add_argument('--input', required=True, help='filename or directory of input image(s)')
    parser.add_argument('--output_dir', required=True, help='directory to write output to')
    args = parser.parse_args()
    output_filenames = test_files(args.model, args.input, args.output_dir)
    for fname in output_filenames:
        print(fname)
