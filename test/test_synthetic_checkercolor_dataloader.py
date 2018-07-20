import os
import sys
import cv2

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append('../')
from pytvision.datasets.colorchecker_dataset import SyntheticColorCheckerExDataset
from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view


def test_colorchecker():

    data = SyntheticColorCheckerExDataset(
            generate=SyntheticColorCheckerExDataset.generate_image_and_annotations,
            transform=transforms.Compose([
                ## resize and crop                           
                mtrans.ToResize( (500,500), resize_mode='asp' ) ,
                #mtrans.CenterCrop( (200,200) ),
                #mtrans.RandomCrop( (255,255), limit=50, padding_mode=cv2.BORDER_REFLECT_101  ),
                #mtrans.ToResizeUNetFoV(388, cv2.BORDER_REFLECT_101),     
                ## color 
                mtrans.RandomSaturation(),
                mtrans.RandomHueSaturationShift(),
                mtrans.RandomHueSaturation(),
                mtrans.RandomRGBShift(),
                mtrans.ToNegative(),
                mtrans.RandomRGBPermutation(),
                mtrans.ToGrayscale(),
                ## blur
                #mtrans.ToRandomTransform( mtrans.ToLinealMotionBlur( lmax=1 ), prob=0.5 ),
                #mtrans.ToRandomTransform( mtrans.ToMotionBlur( ), prob=0.5 ),
                #mtrans.ToRandomTransform( mtrans.ToGaussianBlur(), prob=0.5 ),
                ## geometrical 
                #mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 )
                #mtrans.ToRandomTransform( mtrans.VFlip(), prob=0.5 )
                #mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REFLECT101 ),
                #mtrans.RandomGeometricalTranform( angle=360, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REFLECT101),
                #mtrans.RandomElasticDistort( size_grid=50, padding_mode=cv2.BORDER_REFLECT101 ),
                ## tensor               
                mtrans.ToTensor(),  
                mtrans.ToNormalization(),
                ])
            )

    dataloader = DataLoader(data, batch_size=3, shuffle=True, num_workers=1 )

    sample_batched = data[0]
    print(sample_batched['annotations'].shape)
    sample_batched = data[1]
    print(sample_batched['annotations'].shape)

    # label_batched = []
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['image'].size(),
    #         sample_batched['annotations'].size()
    #         )
        
    #     image = sample_batched['image'][0,:,...]
    #     annotations = sample_batched['annotations'][0,...]

    #     print(image.shape)
    #     print( np.unique(label) )
    #     print(image.min(), image.max())
            
    #     # plt.figure( figsize=(15,15) )
    #     # plt.imshow( image.permute(1,2,0).squeeze()  ) #, cmap='gray' 
    #     # plt.axis('off')
    #     # plt.ioff()
    #     # plt.show()        

    #     # observe 4th batch and stop.
    #     if i_batch == 3: 
    #         break        


test_colorchecker()