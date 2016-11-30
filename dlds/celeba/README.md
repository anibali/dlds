# CelebA

http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

> CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes
> dataset with more than 200K celebrity images.

DLDS currently installs the "Align&Cropped Images" only.

## Structure

DLDS will install the following files for this dataset:

```
celeba
├── classes.txt
└── celeba.h5
```

`celeba.h5` contains the images for the training and test sets. Its
internal structure is as follows:

| Name                          | Type      | Dimensions              |
| ----------------------------- | --------- | ----------------------- |
| `/test/images`                | Byte      | 19962 x 3 x 218 x 178   |
| `/train/images`               | Byte      | 182637 x 3 x 218 x 178  |

If you wish to split the training data back up into the train/val split
specified in the original dataset, take the first 162770 to be the training set,
and the remaining 19867 to be the validation set.

If you want square images that are even more tightly cropped, I recommend
taking a center crop of the center 128x128 pixels.

## Citation

If you use this dataset, please cite the following:

```
Ziwei Liu, Ping Luo, Xiaogang Wang, Xiaoou Tang; Deep Learning Face Attributes in the Wild; ICCV; 2015
```
