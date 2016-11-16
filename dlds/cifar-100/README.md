# CIFAR-100

https://www.cs.toronto.edu/~kriz/cifar.html

> This dataset is just like the CIFAR-10, except it has 100 classes containing
> 600 images each. There are 500 training images and 100 testing images per
> class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each
> image comes with a "fine" label (the class to which it belongs) and a "coarse"
> label (the superclass to which it belongs).

## Structure

DLDS will install the following files for this dataset:

```
cifar-100
├── cifar-100.h5
├── classes.txt
└── superclasses.txt
```

`cifar-100.h5` contains the images and labels for the training and test sets.
Its internal structure is as follows:

| Name                          | Type      | Dimensions            |
| ----------------------------- | --------- | --------------------- |
| `/meta/superclasses`          | Byte      | 100 x 1               |
| `/test/images`                | Byte      | 10000 x 3 x 32 x 32   |
| `/test/labels`                | Byte      | 10000 x 1             |
| `/train/images`               | Byte      | 50000 x 3 x 32 x 32   |
| `/train/labels`               | Byte      | 50000 x 1             |

The tensor stored in `/meta/superclasses` contains a mapping from each class
label to its corresponding superclass label.

`classes.txt` contains names with human semantics for each class label.
Line 1 contains the name for class label 1, line 2 for class label 2, and so
forth.

`superclasses.txt` contains names with human semantics for each superclass
label. Line 1 contains the name for superclass label 1, and so forth.

## Citation

If you use this dataset, please cite the following:

```
Alex Krizhevsky; Learning Multiple Layers of Features from Tiny Images; 2009
```
