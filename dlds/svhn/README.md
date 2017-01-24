# SVHN

http://ufldl.stanford.edu/housenumbers/

> SVHN is a real-world image dataset for developing machine learning and object
> recognition algorithms with minimal requirement on data preprocessing and
> formatting. It can be seen as similar in flavor to MNIST (e.g., the images are
> of small cropped digits), but incorporates an order of magnitude more labeled
> data (over 600,000 digit images) and comes from a significantly harder,
> unsolved, real world problem (recognizing digits and numbers in natural scene
> images). SVHN is obtained from house numbers in Google Street View images.

For non-commercial use only.

## Structure

DLDS will install the following files for this dataset:

```
svhn
├── classes.txt
└── svhn.h5
```

`svhn.h5` contains the images and labels for the training and test sets. Its
internal structure is as follows:

| Name                          | Type      | Dimensions            |
| ----------------------------- | --------- | --------------------- |
| `/test/images`                | Byte      | 26032 x 3 x 32 x 32   |
| `/test/labels`                | Byte      | 26032 x 1             |
| `/train/main/images`          | Byte      | 73257 x 3 x 32 x 32   |
| `/train/main/labels`          | Byte      | 73257 x 1             |
| `/train/extra/images`         | Byte      | 531131 x 3 x 32 x 32  |
| `/train/extra/labels`         | Byte      | 531131 x 1            |

`classes.txt` contains names with human semantics for each class label.
Line 1 contains the name for class label 1, line 2 for class label 2, and so
forth.

## Citation

If you use this dataset, please cite the following:

```
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng; Reading Digits in Natural Images with Unsupervised Feature Learning; 2011
```
