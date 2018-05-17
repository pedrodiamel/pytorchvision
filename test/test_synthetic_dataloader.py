import os
import sys


import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import scipy.misc
import cv2

sys.path.append('../')
from pytvision.datasets.syntheticdata import SyntethicCircleDataset
from pytvision.datasets import imageutl as imutl
from pytvision.datasets import utility as utl
from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view


data = SyntethicCircleDataset(
        count=100,
        generate=SyntethicCircleDataset.generate_image_mask_and_weight,
        imsize=(512,512),
        sigma=0.01,
        bdraw_grid=True,
        transform=transforms.Compose([

              ## resize and crop
                           
              mtrans.ToResize( (400,400), resize_mode='crop' ) ,
              #mtrans.CenterCrop( (200,200) ),
              mtrans.RandomCrop( (255,255), limit=50, padding_mode=cv2.BORDER_REFLECT_101  ),
              #mtrans.ToResizeUNetFoV(388, cv2.BORDER_REFLECT_101),
              
              ## color 

              #mtrans.RandomSaturation(),
              #mtrans.RandomHueSaturationShift(),
              #mtrans.RandomHueSaturation(),
              #mtrans.RandomRGBShift(),
              #mtrans.ToNegative(),
              #mtrans.RandomRGBPermutation(),
              #mtrans.ToGrayscale(),

              ## blur

              #mtrans.ToRandomTransform( mtrans.ToLinealMotionBlur( lmax=1 ), prob=0.5 ),
              #mtrans.ToRandomTransform( mtrans.ToMotionBlur( ), prob=0.5 ),
              mtrans.ToRandomTransform( mtrans.ToGaussianBlur(), prob=0.5 ),
              
              ## geometrical 

              #mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 )
              #mtrans.ToRandomTransform( mtrans.VFlip(), prob=0.5 )
              #mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REFLECT101 ),
              #mtrans.RandomGeometricalTranform( angle=360, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REFLECT101),
              #mtrans.RandomElasticDistort( size_grid=50, padding_mode=cv2.BORDER_REFLECT101 ),
              
              ## tensor 
              
              mtrans.ToTensor(),
              mtrans.RandomElasticTensorDistort( size_grid=10, deform=0.05 ),
              
              ## normalization

              mtrans.ToNormalization(),
              #mtrans.ToWhiteNormalization(),
              #mtrans.ToMeanNormalization(
              #    mean=[0.485, 0.456, 0.406],
              #    std=[0.229, 0.224, 0.225]
              #    ),


            ])
        )

dataloader = DataLoader(data, batch_size=3, shuffle=True, num_workers=1 )

label_batched = []
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['label'].size(),
          sample_batched['weight'].size()    
         )
    
    image_a = sample_batched['image'][0,:,...]
    image_b = sample_batched['image'][1,:,...]
    image_c = sample_batched['image'][2,:,...]

    image = sample_batched['image'][0,0,...]
    label = sample_batched['label'][0,1,...]
    weight = sample_batched['weight'][0,0,...]
    
    print(torch.min(image), torch.max(image), image.shape )
    print(torch.min(label), torch.max(label), image.shape )
    print(torch.min(weight), torch.max(weight), image.shape )

    print(image_a.shape)
    print( np.unique(label) )
    print(image_a.min(), image_a.max())
        
    plt.figure( figsize=(15,15) )
    plt.subplot(131)
    plt.imshow( image_a.permute(1,2,0).squeeze()  ) #, cmap='gray' 
    plt.axis('off')
    plt.ioff()

    plt.subplot(132)
    #plt.imshow( image_b.permute(1,2,0).squeeze() ) 
    plt.imshow( label ) #cmap='gray'
    plt.axis('off')
    plt.ioff()

    plt.subplot(133)
    #plt.imshow( image_c.permute(1,2,0).squeeze()  ) 
    plt.imshow( weight )
    plt.axis('off')

    plt.ioff()       
    plt.show()        

    # observe 4th batch and stop.
    if i_batch == 3: 
        break        

