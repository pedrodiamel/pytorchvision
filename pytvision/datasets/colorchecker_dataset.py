
import os
import numpy as np
import cv2
from sklearn import datasets

import torch
import torch.utils.data as data
import torch.nn.functional

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

import warnings
warnings.filterwarnings("ignore")


class SyntheticColorCheckerExDataset( data.Dataset ):
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
            pathname = os.path.expanduser( pathname )
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
            annotations, labels = self.compute_targets( image, annotations )
            obj = ObjectImageAndAnnotations( image, annotations, labels  )

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


    def compute_targets(self, image, annotations):
        """ Compute target outputs for the network using images and their annotations.
        """
        labels=[]
        boxs=[]
        for ann in annotations:
            x1,y1,x2,y2,c = ann
            boxs.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(int(c))     
        
        return np.stack( boxs, 0 ), np.stack( labels, 0 )


    def encode(self, boxes, labels, max_size=20 ):
        
        num = boxes.shape[0]
        pad_size = max(0, max_size - num)  
        boxes    = torch.nn.functional.pad(boxes,  ( 0, 0, 0, pad_size ), mode='constant', value=-1)
        labels   = torch.nn.functional.pad(labels, ( 0, pad_size), mode='constant', value=-1)

        return boxes, labels
        

    def collate(self, batch):
        '''Pad images and encode targets.
        As for images are of different sizes, we need to pad them to the same size.
        Args:
            batch: (list) of images, cls_targets, loc_targets.
        Returns:
            padded images, stacked cls_targets, stacked loc_targets.
        '''

        imgs   = [x['image'] for x in batch]
        boxes  = [x['annotations'] for x in batch]
        labels = [x['labels'] for x in batch]

        # imgs -> N,C,H,W 
        input_size = np.stack([ img.shape for img in imgs  ], 0).max(0)

        c,h,w, = input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, c, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encode( boxes[i], labels[i], 20)
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        
        return { 
            'image': inputs, 
            'annotations': torch.stack(loc_targets), 
            'labels': torch.stack(cls_targets)        
            }