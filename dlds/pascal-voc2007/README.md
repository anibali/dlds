# PASCAL VOC2007

http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/

> The goal of this challenge is to recognize objects from a number of visual
> object classes in realistic scenes (i.e. not pre-segmented objects). It is
> fundamentally a supervised learning learning problem in that a training set of
> labelled images is provided.

## Structure

DLDS will install the following files for this dataset:

```
pascal-voc2007
├── classes.txt
└── pascal-voc2007.h5
```

`pascal-voc2007.h5` contains the images from the training and test sets, along
with labels and bounding boxes for each object in those images.
Its internal structure is as follows:

| Name                          | Type      | Dimensions            |
| ----------------------------- | --------- | --------------------- |
| `/test/boxes`                 | Long      | 4952 x 50 x 4         |
| `/test/difficult`             | Byte      | 4952 x 50 x 1         |
| `/test/dims`                  | Long      | 4952 x 2              |
| `/test/images`                | Byte      | 4952 x 3 x 500 x 500  |
| `/test/labels`                | Byte      | 4952 x 50 x 1         |
| `/test/objcount`              | Byte      | 4952 x 1              |
| `/test/truncated`             | Byte      | 4952 x 50 x 1         |
| `/train/boxes`                | Long      | 5011 x 50 x 4         |
| `/train/difficult`            | Byte      | 5011 x 50 x 1         |
| `/train/dims`                 | Long      | 5011 x 2              |
| `/train/images`               | Byte      | 5011 x 3 x 500 x 500  |
| `/train/labels`               | Byte      | 5011 x 50 x 1         |
| `/train/objcount`             | Byte      | 5011 x 1              |
| `/train/truncated`            | Byte      | 5011 x 50 x 1         |

`/*/images` contains the images aligned to the top-left and padded to 500x500
pixels with zeros. `/*/dims` contains the original dimensions of the image,
which can be used to remove the padding. `/*/objcount` contains the number of
objects depicted in each image. `/*/{boxes,labels}` contain the bounding boxes
and class labels for each object (only the first `objcount` entries are valid
for each image). Similarly, `/*/{difficult,truncated}` contain boolean values
(0 or 1) indicating whether the object is marked as difficult or truncated.

If you wish to split the training data back up into the train/val split
specified in the original dataset, take the first 2501 to be the training set,
and the remaining 2510 to be the validation set.

`classes.txt` contains names with human semantics for each class label.
Line 1 contains the name for class label 1, line 2 for class label 2, and so
forth.

## Citation

If you use this dataset, please cite the following:

```
M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman; The PASCAL Visual Object Classes Challenge 2007 (VOC2007) Results
```
