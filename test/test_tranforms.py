
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
    image_rgb = cv2.resize( to_rgb(norm(image)), (500,500) )
    mask_rgb  = cv2.resize(to_rgb(norm(mask)),  (500,500) )
    plt.imshow( np.concatenate( (image_rgb, mask_rgb), axis=1 ) )

def tranform_image_performs(image, transform, num_transform=4, bsave=False, bshow=True):
    
    for i in range(num_transform):
        
        obj = ObjectTransform( image )
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



random.seed(1)
ren = ColorCheckerRender()

image_back = to_rgb(np.ones( (500,500,1), dtype=np.uint8 )*128)
image, mask = ren.generate_for_segmentation_mask( image_back, num=5 )  
#obj = ObjectImageAndMaskTransform( image, mask  )

# plt.figure()
# imagesec(image,mask)
# plt.show()

# Transformation
num_transform = 10
bshow=True
bsave=False


## blur + noise
#transform = mtrans.ToLinealMotionBlur( lmax=1 )
#transform = mtrans.ToMotionBlur( ) 
#transform =  mtrans.ToGaussianBlur() 

## color
transform = mtrans.RandomBrightness( factor=0.5 )
# transform = mtrans.RandomBrightnessShift()
# transform = mtrans.RandomContrast()
# transform = mtrans.RandomSaturation()
# transform = mtrans.RandomHueSaturationShift()
# transform = mtrans.RandomHueSaturation()
# transform = mtrans.RandomRGBShift()
# transform = mtrans.RandomGamma()
# transform = mtrans.RandomRGBPermutation()
# transform = mtrans.ToRandomTransform(mtrans.ToGrayscale())
# transform = mtrans.ToRandomTransform(mtrans.ToNegative())
# transform = mtrans.ToRandomTransform(mtrans.CLAHE())








#transform = mtrans.RandomScale(factor=0.5, padding_mode=cv2.BORDER_REFLECT101 )
#transform = mtrans.RandomGeometricalTranform( angle=360, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_CONSTANT)
#transform = mtrans.RandomCrop( (255,255), limit=100, padding_mode=cv2.BORDER_CONSTANT  )


tranform_image_performs(image, transform, num_transform, bsave, bshow)



# for i in range(num_transform):    
#     print(transform)
#     obj_transform = transform( obj )
#     image_o, mask_o = obj.image, obj.mask
#     image_t, mask_t = obj_transform.image, obj_transform.mask    
#     plt.figure( figsize=(8,8) )
#     plt.subplot(211)
#     imagesec(image,image_t)
#     plt.subplot(212)
#     imagesec(mask,mask_t)
#     plt.show()


