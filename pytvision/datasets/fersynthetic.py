import os
import numpy as np
import cv2
import random

import torch
import torch.utils.data as data
import torch.nn.functional


from ..transforms.ferrender import Generator

from pytvision.datasets import imageutl as imutl
from pytvision.datasets import utility
from pytvision.transforms import functional as F


from pytvision.transforms.aumentation import(     
     ObjectImageMaskAndWeightTransform, 
     ObjectImageTransform, 
     ObjectImageAndLabelTransform, 
     ObjectImageAndMaskTransform, 
     ObjectRegressionTransform, 
     ObjectImageAndAnnotations,
     ObjectImageAndMaskMetadataTransform,
    )


import warnings
warnings.filterwarnings("ignore")


class SyntheticFaceDataset( data.Dataset ):
    '''
    Management for Synthetic Face dataset
    '''
    generate_image = 'image'
    generate_image_and_label = 'image_and_label'
    generate_image_and_mask = 'image_and_mask' 


    def __init__(self, 
        data,
        pathnameback=None,
        ext='jpg',
        count=None,
        num_channels=3,
        generate='image_and_label',
        iluminate=True, angle=45, translation=0.3, warp=0.2, factor=0.2,
        transform=None,
        ):
        """Initialization           
        """            
              
        self.data = data
        self.bbackimage = pathnameback != None
        self.databack = None

        if count is None:
            count = len(data)
        
        if self.bbackimage: 
            pathnameback  = os.path.expanduser( pathnameback )            
            self.databack = imutl.imageProvide( pathnameback, ext=ext );   

        self.num_channels = num_channels
        self.generate = generate
        self.ren = Generator( iluminate, angle, translation, warp, factor );
        self.transform = transform 
        self.count=count       
  

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        # read image 
        image, label = self.data[ (idx)%len(self.data)  ]
        image = utility.to_channels(image, self.num_channels)

        # read background 
        if self.bbackimage:
            idxk = random.randint(1, len(self.databack) - 1 )
            back = self.databack[ idxk ] #(idx)%len(self.databack)
            back = F.resize_image(back, 640, 1024, resize_mode='crop', interpolate_mode=cv2.INTER_LINEAR);
            back = utility.to_channels(back, self.num_channels)
        else:
            back = np.ones( (640,1024,3), dtype=np.uint8 )*255
       
        if self.generate == 'image':
            obj = ObjectImageTransform( image  )
        elif self.generate == 'image_and_label':
            _, image, _ = self.ren.generate( image, back )
            image = utility.to_gray( image.astype(np.uint8)  )
            image_t = utility.to_channels(image, self.num_channels)
            image_t = image_t.astype(np.uint8)  
            label = utility.to_one_hot( int(label) , self.data.numclass)            
            obj = ObjectImageAndLabelTransform( image_t, label )  
            
        elif self.generate == 'image_and_mask':            
            _, image, mask = self.ren.generate( image, back )
            image = utility.to_gray( image.astype(np.uint8)  )
            image_t = utility.to_channels(image, self.num_channels)
            image_t = image_t.astype(np.uint8)             
            #print( image_t.shape, image_t.min(), image_t.max(), flush=True )
            #assert(False)            
            mask = mask[:,:,0]
            mask_t = np.zeros( (mask.shape[0], mask.shape[1], 2) )
            mask_t[:,:,0] = (mask == 0).astype( np.uint8 ) # backgraund
            mask_t[:,:,1] = (mask == 1).astype( np.uint8 )
            obj = ObjectImageAndMaskMetadataTransform( image_t, mask_t, np.array([label]) )
        else: 
            assert(False)         

        if self.transform: 
            obj = self.transform( obj )

        return obj.to_dict()
