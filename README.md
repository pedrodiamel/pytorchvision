# PyTVision
[![Documentation Status](https://readthedocs.org/projects/pytorchvision/badge/?version=latest)](https://pytorchvision.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

The pytvision package consists of my datasets, models, and image transformations methods for computer vision projects. This package also containing the synthetic render methods. The backend transforms using opencv.


## Requirements

    pytorch 0.4.1
    git clone https://github.com/pytorch/vision.git
    cd vision
    python setup.py install


## Installation

From source:

    python setup.py build_ext --inplace
    python setup.py install


## Books

- [Example transformation](https://github.com/pedrodiamel/pytorchvision/blob/master/books/example_transforms.ipynb)
- [Example transformation for pytorch](https://github.com/pedrodiamel/pytorchvision/blob/master/books/example_tranforms_pytorch.ipynb)

## Projects

- [Classification](https://github.com/pedrodiamel/pytorch-classification)

## Kaggle Projects

- [kaggle-datasciencebowl-2018](https://github.com/pedrodiamel/kaggle-datasciencebowl-2018)
- [kaggle-imaterialist](https://github.com/pedrodiamel/kaggle-imaterialist)
- [kaggle-tgs-salt-identification](https://github.com/pedrodiamel/kaggle-tgs-salt-identification)

## Documentation
The full documentation is available at [doc](https://pytorchvision.readthedocs.io/en/latest/)

## Dataset

- cars169 
- cub2011
- StanfordOnlineProducts
- fer+

## Models

- inception_v4
- nasnetalarge
- dexpression
- unet (unet11, unetresnet, ...)
- pspnet

## Transforms

- Color transformation
- Geometrical transformation
- Blur + Noise transformation

## Similar projects 

- https://github.com/albu/albumentations


<!-- 

https://www.youtube.com/watch?v=oJsUvBQyHBs

## Building the documentation
1. Go to `docs/` directory
```
cd docs
```
2. Install required libraries
```
pip install -r requirements.txt
```
3. Build html files
```
make html
```
4. Open `_build/html/index.html` in browser. 
-->


