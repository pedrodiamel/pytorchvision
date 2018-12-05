import os
import numpy as np
import cv2
from sklearn import datasets


import warnings
warnings.filterwarnings("ignore")

from ..transforms.ellipserender import  CircleRender, EllipseRender
from ..transforms.aumentation import ObjectImageMaskAndWeightTransform, ObjectImageTransform, ObjectImageAndLabelTransform, ObjectImageAndMaskTransform, ObjectRegressionTransform
from ..transforms import functional as F

from . import imageutl as imutl
from . import weightmaps as wmap
from . import utility


class SyntethicCircleDataset(object):
    '''
    Management for Synthetic Circle dataset
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
            obj = self.transform( obj )

        return obj.to_dict()


