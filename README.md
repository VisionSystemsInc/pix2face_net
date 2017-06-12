# pix2face
### Direct Estimation of 3D Face Pose and Geometry from 2D Images
![](pix2face_teaser.jpg "pix2face_teaser")

## Requirements
* [pytorch](http://pytorch.org)
* [numpy](http://www.numpy.org)
* [scikit-image](http://scikit-image.org)
* Training Data

  You will need three sets of training images: Input, PNCC, and offsets.

  * Input: The input RGB face image.

  * PNCC: "Projected Normalized Coordinate Code", as described in [1]

  * Offsets: 3D offsets from the "mean face" position to the observed 3D position.

\[1\] X. Zhu, Z. Lei, X. Liu, H. Shi, and S. Z. Li, “Face Alignment Across Large Poses: A 3D Solution”, CVPR 2016.

## Training
```
python train.py --input_dir $INPUT_DIR --PNCC_dir $PNCC_DIR --offsets_dir $OFFSETS_DIR \
--val_input_dir $VAL_INPUT_DIR --val_PNCC_dir $VAL_PNCC_DIR --val_offsets_dir $VAL_OFFSETS_DIR \
--output_dir $OUTPUT_DIR
```

## Testing
```
python test.py --model $OUTPUT_DIR/pix2face_unet.pth \
--input <image_or_directory> --output_dir <output_dir>
```

## Demo
See demo.py for an example of a transformation from image --> PNCC + offsets --> 3D Point Cloud.

In order to run the demo, you will need to train the network or download a pre-trained model.

## Contact
Daniel Crispell [dan@visionsystemsinc.com](mailto:dan@visionsystemsinc.com)
