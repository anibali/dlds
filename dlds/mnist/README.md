# MNIST

http://yann.lecun.com/exdb/mnist/

> The MNIST database of handwritten digits, available from this page, has a
> training set of 60,000 examples, and a test set of 10,000 examples. It is a
> subset of a larger set available from NIST. The digits have been
> size-normalized and centered in a fixed-size image.

## Structure

DLDS will install the following files for this dataset:

```
mnist
├── classes.txt
└── mnist.h5
```

`mnist.h5` contains the images and labels for the training and test sets. Its
internal structure is as follows:

| Name              | Type      | Dimensions            |
| ----------------- | --------- | --------------------- |
| `/test/images`    | Byte      | 10000 x 1 x 28 x 28   |
| `/test/labels`    | Byte      | 10000 x 1             |
| `/train/images`   | Byte      | 60000 x 1 x 28 x 28   |
| `/train/labels`   | Byte      | 60000 x 1             |

`classes.txt` contains names with human semantics for each class label.
Line 1 contains the name for class label 1, line 2 for class label 2, and so
forth.

## Citation

If you use this dataset, please cite the following:

```
Y. LeCun, L. Bottou, Y. Bengio, P. Haffner; Gradient-based learning applied to document recognition; Proceedings of the IEEE; November 1998.
```
