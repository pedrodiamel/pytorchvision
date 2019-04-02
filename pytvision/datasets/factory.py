

import os
import numpy as np
from torchvision import datasets 

from . import imaterialist
from . import ferp
from . import fer
from . import cub2011
from . import cars196
from . import stanford_online_products
from . import afew
from . import affect


def create_folder(pathname, name):    
    # create path name dir        
    pathname = os.path.join(pathname, name )
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    return pathname

class FactoryDataset(object):
    
    training = 'train'
    validation = 'val'
    test = 'test'

    mnist='mnist'
    fashion='fashion'
    emnist='emnist'
    cifar10='cifar10'
    cifar100='cifar100'
    stl10='stl10'
    svhn='svhn'

    imaterialist='imaterialist'
    
    ferp='ferp'
    ck='ck'
    jaffe='jaffe'
    bu3dfe='bu3dfe'
    afew='afew'
    affect='affectnet'
    affectdark='affectnetdark'


    cub2011='cub2011'
    cars196='cars196'
    stanford_online_products='stanford_online_products'
    cub2011metric='cub2011metric'
    cars196metric='cars196metric'
    
    @classmethod
    def _checksubset(self, subset): 
        return subset=='train' or subset=='val' or subset=='test'

    @classmethod
    def factory(self,         
        pathname,
        name,
        subset='train',
        download=False,
        transform=None,
        ):
        """Factory dataset
        """

        assert( self._checksubset(subset) )
        pathname = os.path.expanduser(pathname)
        

        # pythorch vision dataset soported

        if name == 'mnist':   
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)
            data = datasets.MNIST( pathname, train=btrain, transform=transform, download=download)       
            data.labels = np.array( data.targets )        

        elif name == 'fashion':
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)
            data = datasets.FashionMNIST(pathname, train=btrain, transform=transform, download=download)
            data.labels = np.array( data.targets )

        elif name == 'emnist':            
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)
            data = datasets.EMNIST(pathname, split='byclass', train=btrain, transform=transform, download=download)
            data.labels = np.array( data.targets )  

        elif name == 'cifar10':     
            btrain=(subset=='train')  
            pathname = create_folder(pathname, name)     
            data = datasets.CIFAR10(pathname, train=btrain, transform=transform, download=download)
            data.labels = np.array( data.targets )  

        elif name == 'cifar100':  
            btrain=(subset=='train')  
            pathname = create_folder(pathname, name)          
            data = datasets.CIFAR100(pathname, train=btrain, transform=transform, download=download)
            data.labels = np.array( data.targets )

        elif name == 'stl10':  
            split= 'train' if (subset=='train')  else 'test'
            pathname = create_folder(pathname, name)          
            data = datasets.STL10(pathname, split=split, transform=transform, download=download)

        elif name == 'svhn':
            split= 'train' if (subset=='train')  else 'test'
            pathname = create_folder(pathname, name)          
            data = datasets.SVHN(pathname, split=split, transform=transform, download=download)
            data.classes = np.unique( data.labels )


        # internet dataset
   
        elif name == 'cub2011':  
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)         
            data = cub2011.CUB2011(pathname, train=btrain,  download=download)
            data.labels = np.array( data.targets )  


        elif name == 'cars196':  
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)         
            data = cars196.Cars196(pathname, train=btrain,  download=download)
            data.labels = np.array( data.targets ) 

        elif name == 'stanford_online_products':  
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)       
            data = stanford_online_products.StanfordOnlineProducts(pathname, train=btrain,  download=download)
            data.labels = np.array( data.targets )  
            data.btrain = btrain


        # kaggle dataset
        elif name == 'imaterialist':
            pathname = create_folder(pathname, name)
            data = imaterialist.IMaterialistDatset(pathname, subset, 'jpg')


        # fer datasets

        elif name == 'ferp':
            pathname = create_folder(pathname, name)            
            if subset=='train':   subfolder = ferp.train 
            elif subset=='val':   subfolder = ferp.valid
            elif subset=='test':  subfolder = ferp.test
            else:                 assert(False)    
            data = ferp.FERPDataset( pathname, subfolder, download=download )

        elif name == 'ck':
            btrain=(subset=='train') 
            pathname = create_folder(pathname, name)
            data = fer.FERClassicDataset(pathname, 'ck', idenselect=[], train=btrain )

        elif name == 'ckp':
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)
            data = fer.FERClassicDataset(pathname, 'ckp', idenselect=[], train=btrain )

        elif name == 'jaffe':
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)
            data = fer.FERClassicDataset(pathname, 'jaffe', idenselect=[], train=btrain )
 
        elif name == 'bu3dfe':
            btrain=(subset=='train')
            pathname = create_folder(pathname, name) 
            data = fer.FERClassicDataset(pathname, 'bu3dfe', idenselect=[], train=btrain )

        elif name == 'afew':  
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)         
            data = afew.Afew(pathname, train=btrain,  download=download)
            data.labels = np.array( data.targets )   
    
        elif name == 'affectnet':
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)
            data = affect.create_affect(path=pathname, train=btrain )

        elif name == 'affectnetdark':
            btrain=(subset=='train')
            pathname = create_folder(pathname, 'affectnet')
            data = affect.create_affectdark(path=pathname, train=btrain )


        # metric learning dataset

        elif name == 'cub2011metric':  
            btrain=(subset=='train')       
            pathname = create_folder(pathname, 'cub2011')
            data = cub2011.CUB2011MetricLearning(pathname, train=btrain,  download=download)
            data.labels = np.array( data.targets )  

        elif name == 'cars196metric':  
            btrain=(subset=='train')       
            pathname = create_folder(pathname, 'cars196')
            data = cars196.Cars196MetricLearning(pathname, train=btrain,  download=download)
            data.labels = np.array( data.targets ) 
            
        else: 
            assert(False)

        data.btrain = (subset=='train')
        return data