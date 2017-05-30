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
├── annot-test.h5
├── annot-train.h5
├── annot-val.h5
└── mpii-human-pose.h5
```

`mpii-human-pose.h5` contains the images for the training, validation, and test
sets. Its internal structure is as follows:

| Name                          | Type      | Dimensions            |
| ----------------------------- | --------- | --------------------- |
| `/test/images`                | Byte      | 11731 x 3 x 550 x 550 |
| `/test/transforms/m`          | Float     | 11731 x 2 x 2         |
| `/test/transforms/b`          | Float     | 11731 x 1 x 2         |
| `/test/normalize`             | Float     | 11731                 |
| `/train/images`               | Byte      | 25925 x 3 x 550 x 550 |
| `/train/parts/coords`         | Float     | 25925 x 16 x 2        |
| `/train/parts/visible`        | Byte      | 25925 x 16            |
| `/train/transforms/m`         | Float     | 25925 x 2 x 2         |
| `/train/transforms/b`         | Float     | 25925 x 1 x 2         |
| `/train/normalize`            | Float     | 25925                 |
| `/val/images`                 | Byte      | 2958 x 3 x 550 x 550  |
| `/val/parts/coords`           | Float     | 2958 x 16 x 2         |
| `/val/parts/visible`          | Byte      | 2958 x 16             |
| `/val/transforms/m`           | Float     | 2958 x 2 x 2          |
| `/val/transforms/b`           | Float     | 2958 x 1 x 2          |
| `/val/normalize`              | Float     | 2958                  |

The center 384x384 pixels of each image contain the human subject. The
surrounding pixels are there for if you want to perform augmentations.

The validation set is the same as Newell et al and Tompson et al.

The `transforms` datasets contain matrices for transforming coordinates from
processed image space to the original MPII image space. This
is useful for evaluation. The formula for restoring coordinates is a simple
linear transform: `orig_coords = proc_coords * M + B`. In Torch this would
be coded as follows (where `proc_coords` is an n x 2 matrix of coordinates
to be transformed):

```lua
local orig_coords = torch.mm(proc_coords, M):add(B:expandAs(proc_coords))
```

The `normalize` datasets contain head segment lengths for each example. These
can be used to calculate PCKh scores.

The other `.h5` files are annotations downloaded verbatim from
https://github.com/anewell/pose-hg-train/tree/4637618a1b162d80436bfd0b557833b5824cbb21/data/mpii/annot.

## Citation

If you use this dataset, please cite the following:

```
M. Andriluka, L. Pishchulin, P. Gehler, B. Schiele; 2D Human Pose Estimation: New Benchmark and State of the Art Analysis; CVPR; 2014
```
