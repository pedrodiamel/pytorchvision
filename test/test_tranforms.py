
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

import scipy.misc
import cv2
import time


sys.path.append('../')
from pytvision.transforms import transforms as mtrans
from pytvision.transforms.render import ColorCheckerRender
from pytvision.transforms.aumentation import ObjectImageMaskAndWeightTransform, ObjectImageTransform, ObjectImageAndMaskTransform, ObjectTransform
from pytvision.datasets.utility import to_rgb


def norm(image):
    image = image.astype( np.float )
    image-=image.min()
    image/=image.max()
    image = (image*255).astype( np.uint8 )
    return image

def imagesec(image, mask):    
    image_rgb = cv2.resize( to_rgb(norm(image)), (500,500), interpolation=cv2.INTER_LANCZOS4 )
    mask_rgb  = cv2.resize(to_rgb(norm(mask)),  (500,500), interpolation=cv2.INTER_LANCZOS4 )
    plt.imshow( np.concatenate( (image_rgb, mask_rgb), axis=1 ) )

def tranform_image_performs(image, transform, num_transform=4, bsave=False, bshow=True, bgrid=False):
    
    for i in range(num_transform):
        
        obj = ObjectTransform( image )
        if bgrid: obj._draw_grid(50,(255,255,255))

        start = time.time()
        obj_transform = transform( obj )
        t = time.time() - start

        print('{} ({}sec)'.format(transform,t) )

        image_o = obj.image
        image_t = obj_transform.image
        
        plt.figure( figsize=(8,8) )
        imagesec(image, image_t)
        plt.title( 'Transform: +[{} ({:0.4f}sec)]'.format(transform, t) )

        if bsave:
            print('save figure ...')
            plt.savefig('../out/image_{}_{}.png'.format(transform, i) )

        if bshow: plt.show()

def tranform_image_and_mask_performs(image, mask, transform, num_transform=4, bsave=False, bshow=True, bgrid=False):
    
    for i in range(num_transform):
        
        obj = ObjectImageAndMaskTransform( image, mask )
        if bgrid: obj._draw_grid(50,(255,255,255))

        start = time.time()
        obj_transform = transform( obj )
        t = time.time() - start

        print('{} ({}sec)'.format(transform,t) )

        image_o, mask_o = obj.image, obj.mask
        image_t, mask_t = obj_transform.image, obj_transform.mask 
        
        plt.figure( figsize=(8,8) )
        plt.subplot(211)
        imagesec(image, image_t)
        plt.title( 'Transform: +[{} ({:0.4f}sec)]'.format(transform, t) )
        plt.subplot(212)        
        imagesec(mask,mask_t)

        if bsave:
            print('save figure ...')
            plt.savefig('../out/image_and_mask_{}_{}.png'.format(transform, i) )

        if bshow: plt.show()


random.seed( 1 )
ren = ColorCheckerRender()

image_back = to_rgb(np.ones( (500,500,1), dtype=np.uint8 )*128)
image, mask = ren.generate_for_segmentation_mask( image_back, num=5 )  
#obj = ObjectImageAndMaskTransform( image, mask  )

# plt.figure()
# imagesec(image,mask)
# plt.show()

# Transformation
num_transform = 1
bshow=True
bsave=False
bgrid=False




## color
transform = mtrans.RandomBrightness( factor=0.5 )
# transform = mtrans.RandomBrightnessShift( factor=0.5 )
# transform = mtrans.RandomContrast( factor=0.3 )
# transform = mtrans.RandomSaturation( factor=0.75 )
# transform = mtrans.RandomHueSaturationShift( factor=0.75 )
# transform = mtrans.RandomHueSaturation( hue_shift_limit=(-5, 5), sat_shift_limit=(-11, 11), val_shift_limit=(-11, 11) )
# transform = mtrans.RandomRGBShift()
# transform = mtrans.RandomGamma( factor=0.75  )
# transform = mtrans.RandomRGBPermutation()
# transform = mtrans.ToRandomTransform(mtrans.ToGrayscale(), prob=0.5)
# transform = mtrans.ToRandomTransform(mtrans.ToNegative(), prob=0.5)
# transform = mtrans.ToRandomTransform(mtrans.CLAHE(), prob=0.5)

## blur + noise
#transform = mtrans.ToLinealMotionBlur( lmax=1 )
#transform = mtrans.ToMotionBlur( ) 
#transform =  mtrans.ToGaussianBlur() 

## geometrical transforms
# transform = mtrans.ToRandomTransform(mtrans.ToResize( (255,255) ), prob=0.85)
# transform = mtrans.ToRandomTransform(mtrans.ToResizeUNetFoV( 388), prob=0.85)
# transform = mtrans.RandomCrop( (255,255), limit=100, padding_mode=cv2.BORDER_CONSTANT  )
# transform = mtrans.RandomScale(factor=0.5, padding_mode=cv2.BORDER_CONSTANT )
# transform = mtrans.ToRandomTransform(mtrans.HFlip(), prob=0.85)
# transform = mtrans.ToRandomTransform(mtrans.Rotate270(), prob=0.85)
# transform = mtrans.RandomGeometricalTranform( angle=360, translation=0.5, warp=0.02, padding_mode=cv2.BORDER_CONSTANT)
# transform = mtrans.RandomElasticDistort( size_grid=50, deform=15, padding_mode=cv2.BORDER_REFLECT_101)


## to tensor
# transform = mtrans.ToRandomTransform(mtrans.ToMeanNormalization(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]), prob=0.5)
# transform = mtrans.ToRandomTransform(mtrans.ToWhiteNormalization(), prob=0.5)
# transform = mtrans.ToRandomTransform(mtrans.ToNormalization(), prob=0.5)


# tranform_image_performs(image, transform, num_transform, bsave, bshow, bgrid)
tranform_image_and_mask_performs(image, mask, transform, num_transform, bsave, bshow, bgrid)





