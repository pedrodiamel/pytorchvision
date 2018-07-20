

import os
import numpy as np
import random
from collections import namedtuple

import torch
from pytvision.datasets import utility 
from pytvision.datasets import imageProvide
from pytvision.transforms.aumentation import ObjectImageAndLabelTransform, ObjectImageTransform


from .anchors import (
    anchor_targets_bbox,
    bbox_transform,
    anchors_for_shape,
    guess_shapes
)


import warnings
warnings.filterwarnings("ignore")


class Dataset( object ):
    """
    Generic dataset
    """

    def __init__(self, 
        data,
        num_channels=1,
        transform=None  
        ):
        """
        Initialization 
        Args:
            @data: dataprovide class
            @num_channels: 
            @tranform: tranform           
        """             
        
        self.data = data
        self.num_channels=num_channels        
        self.transform = transform   
        self.labels = data.labels
        self.classes = np.unique(self.labels) 
        self.numclass = len(self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   

        image, label = self.data[idx]
        image = np.array(image) 
        image = utility.to_channels(image, self.num_channels)        
        label = utility.to_one_hot(label, self.numclass)

        obj = ObjectImageAndLabelTransform( image, label )
        if self.transform: 
            sample = self.transform( obj )
        return obj.to_dict()
    
class ResampleDataset( object ):
    """
    Resample data for generic dataset
    """

    def __init__(self, 
        data,
        num_channels=1,
        count=200,
        transform=None  
        ):
        """
        Initialization   
        data: dataloader class
        tranform: tranform           
        """             
        
        self.num_channels=num_channels
        self.data = data        
        self.transform = transform   
        self.labels = data.labels 
        self.count=count
        
        #self.classes = np.unique(self.labels)
        self.classes, self.frecs = np.unique(self.labels, return_counts=True)
        self.numclass = len(self.classes)
        
        #self.weights = 1-(self.frecs/np.sum(self.frecs))
        self.weights = np.ones( (self.numclass,1) )        
        self.reset(self.weights)
        
        self.labels_index = list()
        for cl in range( self.numclass ):             
            indx = np.where(self.labels==cl)[0]
            self.labels_index.append(indx)            

    
    def reset(self, weights):        
        self.dist_of_classes = np.array(random.choices(self.classes, weights=weights, k=self.count ))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):   
                
        idx = self.dist_of_classes[idx]
        class_index = self.labels_index[idx]
        n =  len(class_index)
        idx = class_index[ random.randint(0,n-1) ]

        image, label = self.data[idx]

        image = np.array(image) 
        image = utility.to_channels(image, self.num_channels)            
        label = utility.to_one_hot(label, self.numclass)

        obj = ObjectImageAndLabelTransform( image, label )
        if self.transform: 
            sample = self.transform( obj )
        return obj.to_dict()

class OD_Dataset( object ):
    """ Abstract generator class.
    """

    def __init__(
        self,
        batch_size=1,
        shuffle_groups=True,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
        compute_anchor_targets=anchor_targets_bbox,
        compute_shapes=guess_shapes,
    ):

        """ Initialize Generator object.

        Args
            batch_size             : The size of the batches to generate.
            shuffle_groups         : If True, shuffles the groups each epoch.
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
            image_max_side         : If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            transform_parameters   : The transform parameters used for data augmentation.
            compute_anchor_targets : Function handler for computing the targets of anchors for an image and its annotations.
            compute_shapes         : Function handler for computing the shapes of the pyramid for a given input.
        """

        self.batch_size             = int(batch_size)
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.compute_anchor_targets = compute_anchor_targets
        self.compute_shapes         = compute_shapes
        self.index = 0
        

    def __len__(self):
        return self.size()

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """ Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def generate_anchors(self, image_shape):
        return anchors_for_shape(image_shape, shapes_callback=self.compute_shapes)

    def compute_targets(self, image, annotations):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = image.shape
        anchors   = self.generate_anchors(max_shape)

        regression = np.empty((anchors.shape[0], 4 + 1), dtype=float)
        labels     = np.empty((anchors.shape[0], self.num_classes() + 1), dtype=float)

        # compute regression targets
        labels[ :, :-1], annotations, labels[:, -1] = self.compute_anchor_targets(
                anchors,
                annotations,
                self.num_classes(),
                mask_shape=image.shape,
            )

        regression[:, :-1] = bbox_transform(anchors, annotations)
        regression[:, -1]  = labels[ :, -1]  # copy the anchor states to the regression batch

        return [regression, labels]



