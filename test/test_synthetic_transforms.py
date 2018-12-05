
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
from torchvision import transforms

from PIL import Image
import imageio
import datetime

import scipy.misc
import cv2
import time

import pytest


sys.path.append('../')
import pytvision
from pytvision.transforms import transforms as mtrans
from pytvision.datasets.ellipse_dataset import SyntethicCircleDataset
from pytvision.transforms.aumentation import ObjectImageMaskAndWeightTransform, ObjectImageTransform, ObjectImageAndMaskTransform
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
        
        obj = ObjectImageTransform( image )
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

def tranform_image_and_mask_performs( data, name, num_transform=4, bsave=False, bshow=True, bgrid=False):
    
    frames = []
    for i in range( min(num_transform, len(data))  ):
        

        start = time.time()
        sample = data[i]
        image, mask, weight = sample['image'], sample['label'], sample['weight']

        image = image.permute(1,2,0).numpy()
        mask  = mask.permute(1,2,0).numpy()
        weight  = weight.permute(1,2,0).numpy()

        # print(image.shape)
        # print(mask.shape)

        t = time.time() - start
        print('frame: {} ({}sec)'.format(i,t) )

        frame = np.concatenate( ( stand(image), stand(mask) ), axis=1 )
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt = 'Frame: {} ({:0.4f}sec)'.format(i, t)
        cv2.putText(frame, txt,(10,15), font, 0.35, (0,0,0), 1, cv2.LINE_AA)
        frames.append( frame )  

        if bshow: 
            plt.figure( figsize=(14,4) )
            plt.imshow( frame )
            plt.title( 'Frame: {} ({:0.4f}sec)'.format(i, t) )
            plt.show()

    if bsave:
        filename = 'rec/{}'.format( name )
        create_gif( filename, frames, duration=0.5)
        print('save: ', filename)

def transform_aug():
    return transforms.Compose([

            #mtrans.HFlip(),
            #mtrans.VFlip(),
            #mtrans.Rotate90(),
            #mtrans.Rotate180(),
            #mtrans.Rotate270(),          

            ## resize and crop                           
            mtrans.ToResize( (600,600), resize_mode='asp', padding_mode=cv2.BORDER_CONSTANT ) ,
            #mtrans.ToPad( 20, 20, padding_mode=cv2.BORDER_CONSTANT ) ,
            #mtrans.CenterCrop( (200,200) ),
            #mtrans.RandomCrop( (400,400), limit=50, padding_mode=cv2.BORDER_REFLECT_101  ),
            #mtrans.ToResizeUNetFoV(388, cv2.BORDER_REFLECT_101),         
            
            # ## color 
            # mtrans.ToRandomChoiceTransform( [
            # mtrans.RandomSaturation(),
            # mtrans.RandomHueSaturationShift(),
            # mtrans.RandomHueSaturation(),
            # mtrans.RandomRGBShift(),
            # #mtrans.ToNegative(),
            # mtrans.RandomRGBPermutation(),
            # mtrans.ToRandomTransform( mtrans.ToGrayscale(), prob=0.5 ),
            # #mtrans.ToGrayscale(),
            # ]),

            ## blur
            # mtrans.ToRandomTransform( mtrans.ToLinealMotionBlur( lmax=1 ), prob=0.5 ),
            mtrans.ToRandomTransform( mtrans.ToMotionBlur( ), prob=1.0 ),
            # mtrans.ToRandomTransform( mtrans.ToGaussianBlur(), prob=0.75 ),
            
            ## geometrical 
            #mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 ),
            #mtrans.ToRandomTransform( mtrans.VFlip(), prob=0.5 ),
            #mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REFLECT101 ),
            #mtrans.RandomGeometricalTransform( angle=360, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REFLECT101),
            #mtrans.RandomElasticDistort( size_grid=50, padding_mode=cv2.BORDER_REFLECT101 ),
            
            
            ## tensor               
            mtrans.ToTensor(),
            # mtrans.RandomElasticTensorDistort( size_grid=10, deform=0.05 ),
            
            ## normalization
            mtrans.ToNormalization(),
            #mtrans.ToWhiteNormalization(),
            #mtrans.ToMeanNormalization(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225]
            #    ),

            ])



def test_image_transform():

    pytvision.transforms.aumentation.interpolate_image_mode = cv2.INTER_LINEAR

    image = cv2.imread( '../rec/image.jpg' )
    image = image[:,:,(2,1,0)] #[:,:,0]

    obj = ObjectImageTransform( image )
    trans = transform_aug()

    start = time.time()
    obj_t = trans(obj)
    t = time.time() - start
    print('Time {}(s)'.format(t))

    image_t = obj_t.image.permute((1,2,0)).numpy()#.squeeze(-1)

    plt.figure(figsize=(8,8))
    plt.imshow(image_t)
    plt.show()



test_image_transform()


def test_transform():

    # Transformation
    num_transform = 3
    bshow=False
    bsave=False
    bgrid=True
    name = 'syntetic_transformations'

    random.seed( 1 )
    data = SyntethicCircleDataset(
            count=300,
            generate=SyntethicCircleDataset.generate_image_mask_and_weight,
            imsize=(512,612),
            sigma=0.01,
            bdraw_grid=bgrid,
            transform=transform_aug()
            )

    tranform_image_and_mask_performs(data, name, num_transform, bsave, bshow, bgrid)


# test_transform()

