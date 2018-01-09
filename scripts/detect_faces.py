import os
import sys
import numpy as np
from PIL import Image
import dlibnet


def main(image_dir, output_dir, detector_model_fname):
    """
    Detect a face in every image in image_dir and write a corresponding output file
    containing the image coordinates of the bounding box in the following form:
    left top right bottom
    """
    detector = dlibnet.mm_object_detector(detector_model_fname)
    for fname in os.listdir(image_dir):
        image_path = os.path.join(image_dir, fname)
        basename, _ = os.path.splitext(fname)
        output_path = os.path.join(output_dir, basename + '_face_box.txt')
        img = np.array(Image.open(image_path))
        dets = detector.detect(img)
        if len(dets) == 0:
            rect = dlibnet.rectangle(0,0,img.shape[1],img.shape[0])
        else:
            dets.sort(key=lambda x : -x.detection_confidence)
            rect = dets[0].rect
        with open(output_path,'w') as fd:
            fd.write('%d %d %d %d\n' % (rect.left, rect.top, rect.right, rect.bottom))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: ' + sys.argv[0] + ' <image_dir> <output_dir> <detector_model_fname>')
        sys.exit(-1)
    image_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_fname = sys.argv[3]
    main(image_dir, output_dir, model_fname)
