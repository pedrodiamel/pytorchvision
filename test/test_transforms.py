
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import Image
import imageio
import datetime

import scipy.misc
import cv2
import time


sys.path.append('../')
from pytvision.transforms import transforms as mtrans
from pytvision.transforms.render import ColorCheckerRender
from pytvision.transforms.aumentation import ObjectImageMaskAndWeightTransform, ObjectImageTransform, ObjectImageAndMaskTransform, ObjectTransform
from pytvision.datasets.utility import to_rgb


def create_gif(pathname, frames, duration=0.2):
    #datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')    
    pathname = '{}.gif'.format( pathname,  )
    imageio.mimsave(pathname, frames, duration=duration)

def norm(image):
    image = image.astype( np.float )
    image-=image.min()
    image/=image.max()
    image = (image*255).astype( np.uint8 )
    return image

def stand( image, imsize=(250,250) ):
    return cv2.resize( to_rgb(norm(image)), imsize, interpolation=cv2.INTER_LANCZOS4 )

def tranform_image_performs(image, transform, num_transform=4, bsave=False, bshow=True, bgrid=False):
    

    frames = []
    for i in range(num_transform):
        
        obj = ObjectTransform( image )
        if bgrid: obj._draw_grid(50,(255,255,255))

        start = time.time()
        obj_transform = transform( obj )
        t = time.time() - start

        print('{} ({}sec)'.format(transform,t) )

        image_o = obj.image
        image_t = obj_transform.image
        frame = np.concatenate( ( stand(image), stand(image_t) ), axis=1 )

        font = cv2.FONT_HERSHEY_SIMPLEX
        txt = 'Transform: + [{} ({:0.4f}sec)]'.format(transform, t)
        cv2.putText(frame, txt,(10,15), font, 0.35,(255,255,255), 1, cv2.LINE_AA)
        frames.append( frame )            

        if bshow: 
            plt.figure( figsize=(8,4) )
            plt.imshow( frame )
            plt.title( 'Transform: +[{} ({:0.4f}sec)]'.format(transform, t) )            
            plt.show()
    
    if bsave:
        filename = '../rec/{}'.format( transform )
        create_gif( filename, frames, duration=0.5)
        print('save: ', filename)

def tranform_image_and_mask_performs(image, mask, transform, num_transform=4, bsave=False, bshow=True, bgrid=False):
    

    frames = []
    for i in range(num_transform):
        
        obj = ObjectImageAndMaskTransform( image, mask )
        if bgrid: obj._draw_grid(50,(255,255,255))

        start = time.time()
        obj_transform = transform( obj )
        t = time.time() - start

        print('{} ({}sec)'.format(transform,t) )

        image_o, mask_o = obj.image, obj.mask
        image_t, mask_t = obj_transform.image, obj_transform.mask 
        frame = np.concatenate( ( stand(image), stand(mask), stand(image_t), stand(mask_t) ), axis=1 )
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt = 'Transform: + [{} ({:0.4f}sec)]'.format(transform, t)
        cv2.putText(frame, txt,(10,15), font, 0.35,(255,255,255), 1, cv2.LINE_AA)
        frames.append( frame )  

        if bshow: 
            plt.figure( figsize=(14,4) )
            plt.imshow( frame )
            plt.title( 'Transform: +[{} ({:0.4f}sec)]'.format(transform, t) )
            plt.show()

    if bsave:
        filename = '../rec/{}'.format( transform )
        create_gif( filename, frames, duration=0.5)
        print('save: ', filename)


random.seed( 1 )
ren = ColorCheckerRender()
image_back = to_rgb(np.ones( (500,500,1), dtype=np.uint8 )*128)
image, mask = ren.generate_for_segmentation_mask( image_back, num=5 )  

# plt.figure()
# imagesec(image,mask)
# plt.show()

# Transformation
num_transform = 1
bshow=True
bsave=False
bgrid=False



## color
# transform = mtrans.RandomBrightness( factor=0.75 )
# transform = mtrans.RandomBrightnessShift( factor=0.5 )
# transform = mtrans.RandomContrast( factor=0.5 )
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
# transform = mtrans.ToLinealMotionBlur( lmax=1 )
# transform = mtrans.ToMotionBlur( ) 
transform =  mtrans.ToGaussianBlur() 

## geometrical transforms
# transform = mtrans.ToRandomTransform(mtrans.ToResize( (255,255) ), prob=0.85)
# transform = mtrans.ToRandomTransform(mtrans.ToResizeUNetFoV( 388), prob=0.85)
# transform = mtrans.RandomCrop( (255,255), limit=100, padding_mode=cv2.BORDER_CONSTANT  )
# transform = mtrans.RandomScale(factor=0.5, padding_mode=cv2.BORDER_CONSTANT )
# transform = mtrans.ToRandomTransform(mtrans.HFlip(), prob=0.85)
# transform = mtrans.ToRandomTransform(mtrans.Rotate270(), prob=0.85)
# transform = mtrans.RandomGeometricalTransform( angle=360, translation=0.5, warp=0.02, padding_mode=cv2.BORDER_CONSTANT)
# transform = mtrans.RandomElasticDistort( size_grid=50, deform=15, padding_mode=cv2.BORDER_REFLECT_101)


## to tensor
# transform = mtrans.ToRandomTransform(mtrans.ToMeanNormalization(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]), prob=0.5)
# transform = mtrans.ToRandomTransform(mtrans.ToWhiteNormalization(), prob=0.5)
# transform = mtrans.ToRandomTransform(mtrans.ToNormalization(), prob=0.5)


#tranform_image_performs(image, transform, num_transform, bsave, bshow, bgrid)
tranform_image_and_mask_performs(image, mask, transform, num_transform, bsave, bshow, bgrid)





