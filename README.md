# Deep learning datasets (DLDS)

The purpose of DLDS is to make fetching and preprocessing datasets an
automatic and painless process.

* Necessary resources are automatically downloaded and checked for integrity.
* Datasets are processed into HDF5 files, which can be read using a variety of
  languages including Lua, Python, and Matlab.
* Class labels all use 1-based indexing

## Configuration

Copy `config.example.json` to `config.json` and customise to your liking.

## Usage

TODO: Dockerize

Example: Install the MNIST data set.

```
th dlds/main.lua mnist
```

## Supported datasets

* [x] [MNIST](dlds/mnist)
* [x] [STL-10](dlds/stl-10)
* [ ] CIFAR-10
* [ ] Places205

## Conventions

* Labels stored in an n x 1 tensor
* Images stored in an n x channels x height x width tensor
