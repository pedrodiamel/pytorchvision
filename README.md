# PyTVision

The pytvision package consists of my datasets, model architectures, and image transformations for computer vision. This package also containing the synthetic render methods

## Installation

From source:

    python setup.py install


## Transforms

The backend transforms using opencv

### Color

transform = mtrans.RandomBrightness( factor=0.75 )





<!-- 
transform = mtrans.RandomBrightnessShift( factor=0.5 )
transform = mtrans.RandomContrast( factor=0.3 )
transform = mtrans.RandomSaturation( factor=0.75 )
transform = mtrans.RandomHueSaturationShift( factor=0.75 )
transform = mtrans.RandomHueSaturation( hue_shift_limit=(-5, 5), sat_shift_limit=(-11, 11), val_shift_limit=(-11, 11) )
transform = mtrans.RandomRGBShift()
transform = mtrans.RandomGamma( factor=0.75  )
transform = mtrans.RandomRGBPermutation()
transform = mtrans.ToRandomTransform(mtrans.ToGrayscale(), prob=0.5)
transform = mtrans.ToRandomTransform(mtrans.ToNegative(), prob=0.5)
transform = mtrans.ToRandomTransform(mtrans.CLAHE(), prob=0.5) 
-->


### Blur + Noise

### Geometrical




