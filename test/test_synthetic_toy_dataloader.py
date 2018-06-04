import os
import sys


import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2


sys.path.append('../')
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytvision.datasets.syntheticdata import ToyDataset
from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view


data = ToyDataset(
        count=500,
        name=ToyDataset.gauss_3d,
        transform=transforms.Compose([              
              ## tensor               
              mtrans.ToTensor(),
            ])
        )

dataloader = DataLoader(data, batch_size=500, shuffle=False, num_workers=1 )

label_batched = []
for i_batch, (x,y) in enumerate(dataloader):
    print(i_batch, x.size(), y.size())
    
    dim = x.shape[1]
    print(dim)

    plt.figure( figsize=(8,8) )
    plt.scatter( x[:,0], x[:,1], c=y, alpha=0.9 )    
    plt.show()      

    # observe 4th batch and stop.
    if i_batch == 0: 
        break        

