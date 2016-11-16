# CIFAR-10

https://www.cs.toronto.edu/~kriz/cifar.html

> The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with
> 6000 images per class. There are 50000 training images and 10000 test images.

## Structure

DLDS will install the following files for this dataset:

```
mnist
├── classes.txt
└── cifar-10.h5
```

`cifar-10.h5` contains the images and labels for the training and test sets. Its
internal structure is as follows:

| Name                          | Type      | Dimensions            |
| ----------------------------- | --------- | --------------------- |
| `/test/images`                | Byte      | 10000 x 3 x 32 x 32   |
| `/test/labels`                | Byte      | 10000 x 1             |
| `/train/images`               | Byte      | 50000 x 3 x 32 x 32   |
| `/train/labels`               | Byte      | 50000 x 1             |

All five data batches are concatenated together to form the training set. If
you wish to separate them out again, the first batch is the first 10000 training
examples, the second is the second 10000, and so forth.

`classes.txt` contains names with human semantics for each class label.
Line 1 contains the name for class label 1, line 2 for class label 2, and so
forth.

## Citation

If you use this dataset, please cite the following:

```
Alex Krizhevsky; Learning Multiple Layers of Features from Tiny Images; 2009
```
