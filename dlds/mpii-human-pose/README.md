# MPII Human Pose

http://human-pose.mpi-inf.mpg.de/

> MPII Human Pose dataset is a state of the art benchmark for evaluation of
> articulated human pose estimation. The dataset includes around 25K images
> containing over 40K people with annotated body joints. The images were
> systematically collected using an established taxonomy of every day human
> activities. Overall the dataset covers 410 human activities and each image is
> provided with an activity label. Each image was extracted from a YouTube video
> and provided with preceding and following un-annotated frames. In addition,
> for the test set we obtained richer annotations including body part occlusions
> and 3D torso and head orientations.

## Structure

DLDS will install the following files for this dataset:

```
mpii-human-pose
├── classes.txt
└── mpii-human-pose.h5
```

`mpii-human-pose.h5` contains the images for the training and test
sets. Its internal structure is as follows:

| Name                          | Type      | Dimensions            |
| ----------------------------- | --------- | --------------------- |
| `/test/images`                | Byte      | 11731 x 3 x 256 x 256 |
| `/test/transforms/m`          | Float     | 11731 x 2 x 2         |
| `/test/transforms/b`          | Float     | 11731 x 2             |
| `/train/images`               | Byte      | 28883 x 3 x 256 x 256 |
| `/train/transforms/m`         | Float     | 28883 x 2 x 2         |
| `/train/transforms/b`         | Float     | 28883 x 2             |
| `/train/parts/coords`         | Float     | 28883 x 16 x 2        |
| `/train/parts/visible`        | Byte      | 28883 x 16            |

The values stored in `transforms` can be used to convert coordinates in
normalized image space (where (-1, -1) is top-left and (1, 1) is bottom-right)
to coordinates in the original coordinate system of the MPII Human Pose dataset.
This is useful if you intend submitting results for evaluation on the test set.

```
origcoords = m * normcoords + b
```

## Citation

If you use this dataset, please cite the following:

```
M. Andriluka, L. Pishchulin, P. Gehler, B. Schiele; 2D Human Pose Estimation: New Benchmark and State of the Art Analysis; CVPR; 2014
```
