

import os
import numpy as np

import torch
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore")

from .imageutl import dsxbExProvide
from ..transforms.aumentation import  ObjectImageMaskAndWeightTransform

train = 'train'
validation = 'val'
test  = 'test'


class DSXBDataset(Dataset):
    '''
    Mnagement for Data Science Bowl image dataset
    https://www.kaggle.com/c/data-science-bowl-2018/data
    '''

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='contours',
        folders_weights='weights',
        ext='png',
        num_channel=3,
        transform=None,
        ):
        """           
        """            
           
        self.data = dsxbExProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                folders_contours,
                folders_weights,
                ext
                )

        self.transform = transform    
        self.num_channel = num_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   

        image, label, contours, weight = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )        
        label_t = np.zeros( (label.shape[0],label.shape[1],3) )
        label_t[:,:,0] = (label < 128)
        label_t[:,:,1] = (label > 128)
        label_t[:,:,2] = (contours > 128)
        weight_t = weight[:,:,np.newaxis]      

        obj = ObjectImageMaskAndWeightTransform( image_t, label_t, weight_t  )
        if self.transform: 
            sample = self.transform( obj )
        return obj.to_output()




class DSXBExDataset(Dataset):
    '''
    Mnagement for Data Science Bowl image dataset
    https://www.kaggle.com/c/data-science-bowl-2018/data
    '''

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='contours',
        folders_weights='weights',
        ext='png',
        transform=None,
        count=1000,
        num_channel=3,
        ):
        """           
        """            
           
        self.data = dsxbExProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                folders_contours,
                folders_weights,
                ext
                )


        self.transform = transform  
        self.count = count  
        self.num_channel = num_channel  

    def __len__(self):
        return self.count  

    def __getitem__(self, idx):   

        idx = len(self.data)%idx
        image, label, contours, weight = self.data[idx] 

        image_t = utility.to_channels(image, ch=self.num_channels )   

        label_t = np.zeros( (label.shape[0],label.shape[1],3) )
        label_t[:,:,0] = (label < 128)
        label_t[:,:,1] = (label > 128)
        label_t[:,:,2] = (contours > 128)

        weight_t = weight[:,:,np.newaxis]        

        obj = ObjectImageMaskAndWeightTransform( image_t, label_t, weight_t  )
        if self.transform: 
            sample = self.transform( obj )
        return obj.to_output()
