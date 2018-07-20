
import os
import numpy as np
import cv2
from sklearn import datasets

from ..transforms.colorchacker_render import ColorCheckerRender
from ..transforms import functional as F

from ..transforms.aumentation import(
     ObjectImageMaskAndWeightTransform, 
     ObjectImageTransform, 
     ObjectImageAndLabelTransform, 
     ObjectImageAndMaskTransform, 
     ObjectRegressionTransform, 
     ObjectImageAndAnnotations
    )

from . import imageutl as imutl
from . import weightmaps as wmap
from . import utility

from .anchors import (
    anchor_targets_bbox,
    bbox_transform,
    anchors_for_shape,
    guess_shapes
)

import warnings
warnings.filterwarnings("ignore")


class SyntheticColorCheckerDataset(object):
    '''
    Management for Synthetic Color Checker dataset
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


        image = self.data[ (idx)%len(self.data)  ]
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
            obj = self.transform( obj )

        return obj.to_dict()

class SyntheticColorCheckerExDataset(object):
    '''
    Management for Synthetic Color Checker dataset
    '''

    generate_image = 'image'
    generate_image_and_mask = 'image_and_mask' 
    generate_image_and_annotations = 'image_and_annotations' 
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
        
        self.num_classes=2
        self.num_channels = 3
        self.generate = generate
        self.bdraw_grid = bdraw_grid
        self.ren = ColorCheckerRender();
        self.transform = transform 
        self.count = count
        self.num=5
  

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        if self.bbackimage:
            image = self.data[ (idx)%len(self.data)  ]
            image = F.resize_image(image, 640, 1024, resize_mode='crop', interpolate_mode=cv2.INTER_LINEAR);
        else:
            image = np.ones( (640,1024,3), dtype=np.uint8 )*255

        #to rgb
        image = utility.to_channels(image, self.num_channels) 

        if self.generate == 'image':
            obj = ObjectImageTransform( image_t  )

        elif self.generate == 'image_and_annotations':  
            image, annotations = self.ren.generate_image_annotations( image, num=self.num )
            annotations = self.compute_targets( image, annotations )

            obj = ObjectImageAndAnnotations( image, np.array(annotations)  )
            print(annotations)

        elif self.generate == 'image_and_mask':            
            image, mask = self.ren.generate_for_segmentation_mask( image, num=self.num )
            image_t = image
            label_t = np.zeros( (mask.shape[0], mask.shape[1], 2) )
            label_t[:,:,0] = (mask <= 0).astype( np.uint8 )
            label_t[:,:,1] = (mask > 0).astype( np.uint8 )
            obj = ObjectImageAndMaskTransform( image_t, label_t  )

        elif self.generate == 'image_mask_and_weight':           
            image, mask = self.ren.generate_for_segmentation_mask( image, num=self.num )   
            weight = wmap.getweightmap( mask )                  
            image_t = image
            label_t = np.zeros( (mask.shape[0], mask.shape[1], 2) )
            label_t[:,:,0] = (mask <= 0).astype( np.uint8 )
            label_t[:,:,1] = (mask > 0).astype( np.uint8 )
            weight_t = weight[:,:,np.newaxis] 
            obj = ObjectImageMaskAndWeightTransform( image_t, label_t, weight_t  )

        else: 
            assert(False) 
        
        if self.bdraw_grid:
            obj._draw_grid( grid_size=50 )

        if self.transform: 
            obj = self.transform( obj )

        return obj.to_dict()


    def generate_anchors(self, image_shape):        
        return anchors_for_shape(image_shape, shapes_callback=guess_shapes)

    def compute_targets(self, image, annotations):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = image.shape
        anchors   = self.generate_anchors(max_shape)

        regression = np.empty((anchors.shape[0], 4 + 1), dtype=float)
        labels     = np.empty((anchors.shape[0], self.num_classes + 1), dtype=float)

        # compute regression targets
        labels[ :, :-1], annotations, labels[:, -1] = self.compute_anchor_targets(
                anchors,
                annotations,
                self.num_classes,
                mask_shape=image.shape,
            )

        regression[:, :-1] = bbox_transform(anchors, annotations)
        regression[:, -1]  = labels[ :, -1]  # copy the anchor states to the regression batch

        return [regression, labels]