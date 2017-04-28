# Stanford Cars

https://ai.stanford.edu/~jkrause/cars/car_dataset.html

> The Cars dataset contains 16,185 images of 196 classes of cars. The data is
> split into 8,144 training images and 8,041 testing images, where each class
> has been split roughly in a 50-50 split. Classes are typically at the level
> of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

## Structure

DLDS will install the following files for this dataset:

```
stanford-cars
├── classes.txt
└── stanford-cars.h5
```

`stanford-cars.h5` contains the images from the training and test sets, along
with labels and bounding boxes for each object in those images.
Its internal structure is as follows:

| Name                          | Type      | Dimensions            |
| ----------------------------- | --------- | --------------------- |
| `/test/boxes`                 | Float     | 8041 x 4              |
| `/test/dims`                  | Float     | 8041 x 2              |
| `/test/images`                | Byte      | 8041 x 3 x 500 x 500  |
| `/test/labels`                | Long      | 8041 x 1              |
| `/train/boxes`                | Float     | 8144 x 4              |
| `/train/dims`                 | Float     | 8144 x 2              |
| `/train/images`               | Byte      | 8144 x 3 x 500 x 500  |
| `/train/labels`               | Long      | 8144 x 1              |

`/*/images` contains the images aligned to the top-left and padded to 500x500
pixels with zeros. `/*/dims` contains the original dimensions of the image,
which can be used to remove the padding. `/*/{boxes,labels}` contain the
bounding boxes and class labels for each image.

`classes.txt` contains names with human semantics for each class label.
Line 1 contains the name for class label 1, line 2 for class label 2, and so
forth.

## Citation

If you use this dataset, please cite the following:

```
J. Krause, M. Stark, J. Deng, L. Fei-Fei; 3D Object Representations for Fine-Grained Categorization; ICCV; 2013
```
