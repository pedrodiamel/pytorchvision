# PyTVision

The pytvision package consists of my datasets, model architectures, and image transformations for computer vision. This package also containing the synthetic render methods

## Installation

From source:

    python setup.py install


## Transforms

The backend transforms using opencv


### Color tranformations

##### Brightness

    transform = mtrans.RandomBrightness( factor=0.75 )
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

Examples:

![Brightness](rec/RandomBrightness.gif)
![Contrast](rec/RandomContrast.gif)
![Saturation](rec/RandomSaturation.gif)


### Blur + Noise

    transform = mtrans.ToLinealMotionBlur( lmax=1 )
    transform = mtrans.ToMotionBlur( ) 
    transform = mtrans.ToGaussianBlur() 

Examples:

![Motion Blur](rec/ToMotionBlur.gif)


### Geometrical transformation

    transform = mtrans.RandomElasticDistort( size_grid=50, deform=15, padding_mode=cv2.BORDER_REFLECT_101)
    transform = mtrans.ToRandomTransform(mtrans.ToResize( (255,255) ), prob=0.85)
    transform = mtrans.ToRandomTransform(mtrans.ToResizeUNetFoV( 388), prob=0.85)
    transform = mtrans.RandomCrop( (255,255), limit=100, padding_mode=cv2.BORDER_CONSTANT  )
    transform = mtrans.RandomScale(factor=0.5, padding_mode=cv2.BORDER_CONSTANT )
    transform = mtrans.ToRandomTransform(mtrans.HFlip(), prob=0.85)
    transform = mtrans.ToRandomTransform(mtrans.Rotate270(), prob=0.85)
    transform = mtrans.RandomGeometricalTranform( angle=360, translation=0.5, warp=0.02, padding_mode=cv2.BORDER_CONSTANT)

Examples:

![Elastic Distort](rec/RandomElasticDistort.gif)
![Geometrical Transform](rec/RandomGeometricalTransform.gif)
![Scale](rec/RandomScale.gif)





