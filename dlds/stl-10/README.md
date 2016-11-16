# STL-10

http://cs.stanford.edu/~acoates/stl10

> The STL-10 dataset is an image recognition dataset for developing unsupervised
> feature learning, deep learning, self-taught learning algorithms. It is
> inspired by the CIFAR-10 dataset but with some modifications. In particular,
> each class has fewer labeled training examples than in CIFAR-10, but a very
> large set of unlabeled examples is provided to learn image models prior to
> supervised training. The primary challenge is to make use of the unlabeled
> data (which comes from a similar but different distribution from the labeled
> data) to build a useful prior. We also expect that the higher resolution of
> this dataset (96x96) will make it a challenging benchmark for developing more
> scalable unsupervised learning methods.

## Structure

DLDS will install the following files for this dataset:

```
stl-10
├── classes.txt
└── stl-10.h5
```

`stl-10.h5` contains the images and labels for the training and test sets. Its
internal structure is as follows:

| Name                          | Type      | Dimensions            |
| ----------------------------- | --------- | --------------------- |
| `/test/images`                | Byte      | 8000 x 3 x 96 x 96    |
| `/test/labels`                | Byte      | 8000 x 1              |
| `/train/labeled/images`       | Byte      | 5000 x 3 x 96 x 96    |
| `/train/labeled/labels`       | Byte      | 5000 x 1              |
| `/train/labeled/fold_indices` | Byte      | 10 x 1000             |
| `/train/unlabeled/images`     | Byte      | 100000 x 3 x 96 x 96  |

`classes.txt` contains names with human semantics for each class label.
Line 1 contains the name for class label 1, line 2 for class label 2, and so
forth.

## Citation

If you use this dataset, please cite the following:

```
Adam Coates, Honglak Lee, Andrew Y. Ng; An Analysis of Single Layer Networks in Unsupervised Feature Learning; AISTATS; 2011
```
