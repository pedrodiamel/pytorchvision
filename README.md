# PyTVision
[![Documentation Status](https://readthedocs.org/projects/pytorchvision/badge/?version=latest)](https://pytorchvision.readthedocs.io/en/latest/?badge=latest)

The pytvision package consists of my datasets, models, and image transformations methods for computer vision projects. This package also containing the synthetic render methods. The backend transforms using opencv.

## Installation

From source:

    python setup.py build_ext --inplace
    python setup.py install

## Books

- [Example transformation](https://github.com/pedrodiamel/pytorchvision/blob/master/books/example_transforms.ipynb)
- [Example transformation for pytorch](https://github.com/pedrodiamel/pytorchvision/blob/master/books/example_tranforms_pytorch.ipynb)


## Projects

- [kaggle-datasciencebowl-2018](https://github.com/pedrodiamel/kaggle-datasciencebowl-2018)
- [kaggle-imaterialist](https://github.com/pedrodiamel/kaggle-imaterialist)
- kaggle-tgs-salt-identification

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

- Geometrical transformation
- Color transformation
- Blur + Noise transformation



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





