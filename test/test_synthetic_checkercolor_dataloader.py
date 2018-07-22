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
            pathname='~/.datasets/real/',
            count=50,
            generate=SyntheticColorCheckerExDataset.generate_image_and_annotations,
            transform=transforms.Compose([
                ## resize and crop                           
                mtrans.ToResize( (224,224), resize_mode='square' ) ,
                #mtrans.CenterCrop( (200,200) ),
                #mtrans.RandomCrop( (255,255), limit=50, padding_mode=cv2.BORDER_REFLECT_101  ),
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

    dataloader = DataLoader(data, batch_size=4, shuffle=False, num_workers=1, collate_fn=data.collate )

    label_batched = []
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['annotations'].size(),
            sample_batched['labels'].size()
            )
        
        image = sample_batched['image'][0, ... ]
        boxs = sample_batched['annotations'][0, ... ]
        labels = sample_batched['labels'][0, ... ]

        index = labels != -1
        boxs = boxs[index, ...]
        labels = labels[index]

        

        # img = image.permute(1,2,0).squeeze().numpy()  * 255      
        # for i, box in enumerate( boxs ):
        #     color = view.STANDARD_COLORS[i%10];
        #     x1, y1, x2, y2 = box
        #     bbox = np.array([ [x1,y1], [x2,y2] ])            
        #     classe = int(labels[i].numpy())
        #     view.draw_bounding_box_dic( img, {'bbox':bbox, 'classe': classe, 'score':100.0 }, color=color, thickness=4 )

        # plt.figure( figsize=(15,15) )
        # plt.imshow( img/255  ) #, cmap='gray' 
        # plt.axis('off')
        # plt.ioff()
        # plt.show()        

        # observe 4th batch and stop.
        # if i_batch == 0: 
        #     break        


test_colorchecker()