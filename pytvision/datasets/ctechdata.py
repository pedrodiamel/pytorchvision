
import os
import numpy as np
from collections import namedtuple
from skimage import io, transform

import warnings
warnings.filterwarnings("ignore")

from . import utility 
from .imageutl import  ctechProvide
from ..transforms.aumentation import  ObjectImageAndMaskTransform


train = 'train'
validation = 'val'
test  = 'test'



class CTECHDataset(object):
    '''
    Management for Caltech dataset
    '''

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        ext='png',
        transform=None,
        count=1000, 
        num_channels=3,
        ):
        """Initialization       
        """            
           
        self.data = ctechProvide.create( 
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                )
        
        self.transform = transform  
        self.count = count    
        self.num_channels = num_channels

    def __len__(self):
        return self.count

    def __getitem__(self, idx):   

        idx = idx%len(self.data)
        image, mask = self.data[idx] 

        image_t = utility.to_channels(image, ch=self.num_channels )
        label_t = np.zeros( (mask.shape[0], mask.shape[1], 2) )
        label_t[:,:,0] = (mask < 0)
        label_t[:,:,1] = (mask > 0)

        obj = ObjectImageAndMaskTransform( image_t, label_t  )
        if self.transform: 
            sample = self.transform( obj )

        return obj.to_output()