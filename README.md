# Deep learning datasets (DLDS)

The purpose of DLDS is to make fetching and preprocessing datasets an
automatic and painless process.

* Necessary resources are automatically downloaded and checked for integrity.
* Datasets are processed into HDF5 files, which can be read using a variety of
  languages including Lua, Python, and Matlab.
* Class labels all use 1-based indexing

## Building

1. Copy `config.example.json` to `config.json` and customise to your liking
3. [Install Docker](https://www.docker.com/products/overview#/install_the_platform)
2. Build the DLDS Docker image with `docker build -t dlds $PWD`

## Usage

Example: Installing the MNIST data set.

```
docker run --rm -it --volume=/data:/data dlds install mnist
```

Ensure that you set the volume(s) to match your particular `config.json`. The
command shown here works with the example config file.

## Supported datasets

* [x] [MNIST](dlds/mnist)
* [x] [STL-10](dlds/stl-10)
* [ ] CIFAR-10
* [ ] Places205

## Conventions

* Labels stored in an n x 1 tensor
* Images stored in an n x channels x height x width tensor
