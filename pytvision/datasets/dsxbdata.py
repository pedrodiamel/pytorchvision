

import os
import numpy as np

import torch
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore")

from . import imageutl as imutl

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
        transform=None,
        ):
        """           
        """            
           
        self.data = imutl.dsxbExProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                folders_contours,
                folders_weights,
                ext
                )

        self.transform = transform      

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   

        image, label, contours, weight = self.data[idx] 

        #to rgb
        if len(image.shape)==2 or (image.shape==3 and image.shape[2]==1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_t = image

        label_t = np.zeros( (label.shape[0],label.shape[1],3) )
        label_t[:,:,0] = (label < 128)
        label_t[:,:,1] = (label > 128)
        label_t[:,:,2] = (contours > 128)

        # label_t = np.zeros( (label.shape[0],label.shape[1]) )
        # label_t[(label > 128)]  = 1    #back, forg
        # label_t[(contours > 128)] = 2
        # label_t = label_t[:,:,np.newaxis]

        weight_t = weight[:,:,np.newaxis]        

        sample = {'image': image_t, 'label':label_t, 'weight':weight_t }
        if self.transform: 
            sample = self.transform(sample)
        
        return sample


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
        count=1000
        ):
        """           
        """            
           
        self.data = imutl.dsxbExProvide(
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

    def __len__(self):
        return self.count #len(self.data)

    def __getitem__(self, idx):   

        idx = len(self.data)%idx
        image, label, contours, weight = self.data[idx] 

        #to rgb
        if len(image.shape)==2 or (image.shape==3 and image.shape[2]==1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_t = image

        label_t = np.zeros( (label.shape[0],label.shape[1],3) )
        label_t[:,:,0] = (label < 128)
        label_t[:,:,1] = (label > 128)
        label_t[:,:,2] = (contours > 128)

        # label_t = np.zeros( (label.shape[0],label.shape[1]) )
        # label_t[(label > 128)]  = 1    #back, forg
        # label_t[(contours > 128)] = 2
        # label_t = label_t[:,:,np.newaxis]

        weight_t = weight[:,:,np.newaxis]        

        sample = {'image': image_t, 'label':label_t, 'weight':weight_t }
        if self.transform: 
            sample = self.transform(sample)
        return sample
