import os

# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms, utils

import numpy as np
import cv2

import warnings
warnings.filterwarnings("ignore")

from ..transforms.render import ColorCheckerRender, CircleRender, EllipseRender
from ..transforms.aumentation import ObjectImageMaskAndWeightTransform, ObjectImageTransform, ObjectImageAndLabelTransform, ObjectImageAndMaskTransform
from ..transforms import functional as F

from . import imageutl as imutl
from . import weightmaps as wmap
from . import utility



class SyntheticColorCheckerDataset(object):
    '''
    Mnagement for Synthetic Color Checker dataset
    '''

    generate_image = 'image'
    generate_image_and_mask = 'image_and_mask' 
    generate_image_mask_and_weight = 'image_mask_and_weight' 

    def __init__(self, 
        pathname,
        ext='jpg',
        bdraw_grid=False,
        generate='image_mask_and_weight',
        transform=None,
        ):
        """Initialization           
        """            

        self.data = imutl.imageProvide(pathname, ext=ext);
        self.num_channels = 3
        self.generate = generate
        self.bdraw_grid = bdraw_grid
        self.ren = ColorCheckerRender();
        self.transform = transform      

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   


        image = self.data[ (dx)%len(self.data)  ]
        image = F.resize_image(image, 640, 1024, resize_mode='crop', interpolate_mode=cv2.INTER_LINEAR);

        image = utility.to_channels(image, self.num_channels)   
        image, mask = self.ren.generate_for_segmentation_mask( image, num=5 )   
        weight = wmap.getweightmap( mask )     

        image_t = image
        label_t = np.zeros( (mask.shape[0], mask.shape[1], 2 ) )
        label_t[:,:,0] = (mask <= 0).astype( np.uint8 )
        label_t[:,:,1] = (mask > 0).astype( np.uint8 )
        weight_t = weight[:,:,np.newaxis] 

        if self.generate == 'image':
            obj = ObjectImageTransform( image_t  )
        elif self.generate == 'image_and_mask':
            obj = ObjectImageAndMaskTransform( image_t, label_t  )
        elif self.generate == 'image_mask_and_weight':
            obj = ObjectImageMaskAndWeightTransform( image_t, label_t, weight_t  )
        else: 
            assert(False) 
        
        if self.bdraw_grid:
            obj._draw_grid( grid_size=50 )

        if self.transform: 
            sample = self.transform( obj )

        return obj.to_dict()



class SyntheticColorCheckerExDataset(object):
    '''
    Mnagement for Synthetic Color Checker dataset
    '''

    generate_image = 'image'
    generate_image_and_mask = 'image_and_mask' 
    generate_image_mask_and_weight = 'image_mask_and_weight' 

    def __init__(self, 
        pathname=None,
        ext='jpg',
        count=100,
        bdraw_grid=False,
        generate='image_mask_and_weight',
        transform=None,
        ):
        """Initialization           
        """            
        
        self.bbackimage = pathname != None
        self.data = None
        if self.bbackimage:
            self.data = imutl.imageProvide(pathname, ext=ext);   
        self.num_channels = 3
        self.generate = generate
        self.bdraw_grid = bdraw_grid
        self.ren = ColorCheckerRender();
        self.transform = transform 
        self.count = count
  

    def __len__(self):
        return self.count

    def __getitem__(self, idx):   


        if self.bbackimage:
            image = self.data[ (dx)%len(self.data)  ]
            image = F.resize_image(image, 640, 1024, resize_mode='crop', interpolate_mode=cv2.INTER_LINEAR);
        else:
            image = np.ones( (640,1024,3), dtype=np.uint8 )*255


        #to rgb
        image = utility.to_channels(image, self.num_channels) 
        image, mask = self.ren.generate_for_segmentation_mask( image, num=5 )   
        weight = wmap.getweightmap( mask )            
        
        image_t = image
        label_t = np.zeros( (mask.shape[0], mask.shape[1], 2) )
        label_t[:,:,0] = (mask <= 0).astype( np.uint8 )
        label_t[:,:,1] = (mask > 0).astype( np.uint8 )
        weight_t = weight[:,:,np.newaxis] 


        if self.generate == 'image':
            obj = ObjectImageTransform( image_t  )
        elif self.generate == 'image_and_mask':
            obj = ObjectImageAndMaskTransform( image_t, label_t  )
        elif self.generate == 'image_mask_and_weight':
            obj = ObjectImageMaskAndWeightTransform( image_t, label_t, weight_t  )
        else: 
            assert(False) 
        
        if self.bdraw_grid:
            obj._draw_grid( grid_size=50 )

        if self.transform: 
            sample = self.transform( obj )

        return obj.to_dict()

class SyntethicCircleDataset(object):
    '''
    Mnagement for Synthetic Circle dataset
    '''

    generate_image = 'image'
    generate_image_and_mask = 'image_and_mask' 
    generate_image_mask_and_weight = 'image_mask_and_weight' 


    def __init__(self, 
        count=100,
        generate='image',
        cnt=5,
        imsize=(512, 512),
        rmin=5, rmax=50,
        border=90,
        sigma=0.2,
        btouch=True,
        bdraw_grid=False,
        transform=None,
        ):
        """Initialization
        Args:
            @count: for batch size         
            @generate
        """            
        
        self.ren = CircleRender();

        self.count = count
        self.imsize= imsize
        self.cnt = cnt
        self.rmin = rmin
        self.rmax = rmax
        self.border = border
        self.sigma = sigma
        self.btouch = btouch
        self.transform = transform 
        self.bdraw_grid = bdraw_grid
        self.generate = generate
    

    def __len__(self):
        return self.count

    def __getitem__(self, idx):   
        
        image, masks, meta = self.ren.generate( 
            self.imsize[0], self.imsize[1], self.cnt, self.rmin, 
            self.rmax, self.border, self.sigma, 
            self.btouch
            )

        edges = utility.get_edges( masks )
        btouchs = utility.get_touchs( edges )  
        bmask  = utility.tolabel(masks)

        #weight = wmap.getweightmap( mask )     
        weight = wmap.getunetweightmap(bmask, masks )
        
        h,w = image.shape[:2]
        image_t = image
        label_t = np.zeros( (h,w, 3) )
        label_t[:,:,0] = (bmask <= 0).astype( np.uint8 )
        label_t[:,:,1] = (bmask > 0).astype( np.uint8 )
        label_t[:,:,2] = (btouchs > 0).astype( np.uint8 )
        weight_t = weight

        #label_t = label_t[:,:,np.newaxis] 
        weight_t = weight[:,:,np.newaxis] 

        if self.generate == 'image':
            obj = ObjectImageTransform( image_t  )
        elif self.generate == 'image_and_mask':
            obj = ObjectImageAndMaskTransform( image_t, label_t  )
        elif self.generate == 'image_mask_and_weight':
            obj = ObjectImageMaskAndWeightTransform( image_t, label_t, weight_t  )
        else: 
            assert(False) 
        
        if self.bdraw_grid:
            obj._draw_grid( grid_size=50 )

        if self.transform: 
            sample = self.transform( obj )

        return obj.to_dict()

