
import os
import numpy as np
import PIL.Image
import cv2
import random
import csv
import pandas as pd


from . import utility as utl

trainfile='stage1_train'
testfile='stage1_test'
testfilefinal='stage2_test_final'


class dataProvide(object):
    
    def __init__(self):
        self.index = 0
        self.data  = []

    def __len__(self):
        return len(self.data)
        
    def _loadimage(self, pathname):
        '''
        Load image using pathname
        '''

        if os.path.exists(pathname):
            try:
                image = PIL.Image.open(pathname)
                image.load()
            except IOError as e:
                raise ValueError('IOError: Trying to load "%s": %s' % (pathname, e.message) ) 
        else:
            raise ValueError('"%s" not found' % pathname)

        if image.mode in ['L', 'RGB']:
            # No conversion necessary
            return image
        elif image.mode in ['1']:
            # Easy conversion to L
            return image.convert('L')
        elif image.mode in ['LA']:
            # Deal with transparencies
            new = PIL.Image.new('L', image.size, 255)
            new.paste(image, mask=image.convert('RGBA'))
            return new
        elif image.mode in ['CMYK', 'YCbCr']:
            # Easy conversion to RGB
            return image.convert('RGB')
        elif image.mode in ['P', 'RGBA']:
            # Deal with transparencies
            new = PIL.Image.new('RGB', image.size, (255, 255, 255))
            new.paste(image, mask=image.convert('RGBA'))
            return new
        else:
            raise ValueError('Image mode "%s" not supported' % image.mode);
        
        return  image;
    

class dsxbExProvide(dataProvide):
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
        ):
        super(dsxbExProvide, self).__init__( );

        self.path = base_folder
        self.subpath = sub_folder
        self.folders_images = folders_images
        self.folders_labels = folders_labels
        self.folders_contours = folders_contours
        self.folders_weights = folders_weights

        self.pathimages   = os.path.join( base_folder, sub_folder, folders_images   )
        self.pathlabels   = os.path.join( base_folder, sub_folder, folders_labels   )
        self.pathcontours = os.path.join( base_folder, sub_folder, folders_contours )
        self.pathweights  = os.path.join( base_folder, sub_folder, folders_weights  )

        self.data = [             
            (
                f, 
                os.path.join(self.pathimages,   '{}'.format(f) ),
                os.path.join(self.pathlabels,   '{}'.format(f) ),
                os.path.join(self.pathcontours, '{}'.format(f) ),
                os.path.join(self.pathweights,  '{}.{}'.format(f.split('.')[0],'txt') ),
            )
            for f in sorted(os.listdir(self.pathimages)) if f.split('.')[-1] == ext             
            ];

    def getid(self): return self.data[self.index][0]


    def __getitem__(self, i):
                
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;                 
        
        #load image
        image_pathname = self.data[i][1]; 
        image = np.array(self._loadimage(image_pathname), dtype=np.uint8)
        
        #load label
        label_pathname = self.data[i][2]; 
        label = np.array(self._loadimage(label_pathname), dtype=np.uint8)

        #load contours
        contours_pathname = self.data[i][3]; 
        contours = np.array(self._loadimage(contours_pathname), dtype=np.uint8)

        # load weights
        weight_pathname = self.data[i][4]; 
        weight = np.loadtxt(weight_pathname, delimiter=",")
        
        return image, label, contours, weight



class dsxbProvide(dataProvide):
    '''
    Mnagement for Data Science Bowl image dataset
    https://www.kaggle.com/c/data-science-bowl-2018/data
    '''

    @classmethod
    def create(
        cls, 
        base_folder,
        sub_folder, 
        id_file_name,
        folders_image='images',
        folders_masks='masks',
        train=True,
        ):
        '''
        Factory function that create an instance of ferProvide and load the data form disk.
        '''
        provide = cls(base_folder, sub_folder, id_file_name ,folders_image, folders_masks, train=True)
        provide.load_folders();
        return provide

    
    def __init__(self,
        base_folder,    
        sub_folder,     
        id_file_name,
        folders_image='images',
        folders_masks='masks',
        train=True
        ):
        super(dsxbProvide, self).__init__( );

        self.base_folder     = base_folder
        self.sub_folders     = sub_folder
        self.folders_image   = folders_image
        self.folders_masks   = folders_masks
        self.id_file_name    = id_file_name
        self.train=train


    def getimage(self,i):
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;
        image_pathname = self.data[i][1];
        #load image
        image = np.array(self._loadimage(image_pathname), dtype=np.uint8)
        return image

    def getlabel(self,i):
        
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');

        self.index = i;
        masks_pathname = self.data[i][2]; 

        #load mask
        label = np.zeros( (image.shape[0], image.shape[1], len(masks_pathname)  ) )
        for k,pathname in enumerate(masks_pathname):
            mask = np.array(self._loadimage(pathname), dtype=np.uint8)
            label[:,:,k] = mask*(k+1) 
        
        return label

    def getid(self): return self.data[self.index][0]
    def getrl(self): return self.data[self.index][3]

    def __getitem__(self, i):
                
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;
                 
        
        #load image
        image_pathname = self.data[i][1]; 
        image = np.array(self._loadimage(image_pathname), dtype=np.uint8)
        
        #load mask
        masks_pathname = self.data[i][2]; 
        label = np.zeros( (image.shape[0], image.shape[1], len(masks_pathname)  ) )
        for k,pathname in enumerate(masks_pathname):
            mask = np.array(self._loadimage(pathname), dtype=np.uint8)
            if len(mask.shape) == 3: mask = mask.max(2)
            label[:,:,k] = (mask>128)*(k+1) 
        
        return image, label

            

    def load_folders(self):
        '''
        load file patch for disk
        '''        
        self.data = []                
        folder_path = os.path.join(self.base_folder, self.sub_folders )
        in_id_path = os.path.join(self.base_folder, self.id_file_name )
        ext = 'png'
        
        with open(in_id_path) as csvfile:            
            id_files = csv.reader(csvfile) 
            head = True
            id_current = ''
            data = {}
            for row in id_files: 
                
                if head: head=False; continue;
                id_data =  row[0]
                rl_code =  [int(x) for x in row[1].split(' ')] 

                # create data
                if id_data != id_current:                    
                    id_current = id_data  
                    # get image
                    images_path = os.path.join(folder_path, id_data, self.folders_image)
                    image_file = os.path.join(images_path, '{}.{}'.format(id_data,ext) )                    
                    # get label
                    masks_path = os.path.join(folder_path, id_data, self.folders_masks) 
                    masks_files = [ os.path.join(masks_path,f) for f in sorted(os.listdir(masks_path)) if f.split('.')[-1] == ext ];
                    # create tupla                   
                    data[id_current] = (id_current, image_file, masks_files, [], [])     

                # add run len code          
                data[id_current][3].append(rl_code)

        # to array        
        data = sorted(data.items())
        self.data = [ v for k,v in data ]



class dsxbImageProvide(dataProvide):
    '''
    Mnagement for Data Science Bowl image dataset
    https://www.kaggle.com/c/data-science-bowl-2018/data
    '''

    @classmethod
    def create(
        cls, 
        base_folder,
        sub_folder, 
        folders_image='images',
        ext = 'png',
        ):
        '''
        Factory function that create an instance of ferProvide and load the data form disk.
        '''
        provide = cls(base_folder, sub_folder ,folders_image, ext)
        provide.load_folders();
        return provide

    
    def __init__(self,
        base_folder,    
        sub_folder,     
        folders_image='images',
        ext = 'png',
        ):
        super(dsxbImageProvide, self).__init__( );
        
        self.base_folder     = base_folder
        self.sub_folders     = sub_folder
        self.folders_image   = folders_image
        self.ext = ext

    def getimage(self,i):
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;
        image_pathname = self.data[i][1];
        #load image
        image = np.array(self._loadimage(image_pathname), dtype=np.uint8)
        return image

    def getid(self): return self.data[self.index][0]
    

    def __getitem__(self, i):
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;
        #load image
        image_pathname = self.data[i][1]; 
        image = np.array(self._loadimage(image_pathname), dtype=np.uint8)               
        return image

            

    def load_folders(self):
        '''
        load file patch for disk
        '''
        self.data = []                
        folder_path = os.path.join(self.base_folder, self.sub_folders )
        data = [ (f,os.path.join(folder_path, f, self.folders_image,'{}.{}'.format(f, self.ext))) for f in sorted(os.listdir(folder_path)) ];
        self.data = data




class ctechProvide(dataProvide):
    '''
    Mnagement for Caltech dataset
    '''

    @classmethod
    def create(
        cls, 
        base_folder,
        sub_folder, 
        folders_image='images',
        folders_label='labels',
        ):
        '''
        Factory function that create an instance of ctechProvide and load the data form disk.
        '''
        provide = cls(base_folder, sub_folder ,folders_image, folders_label)
        provide.load_folders();
        return provide

    
    def __init__(self,
        base_folder,    
        sub_folder,     
        folders_image = 'images',
        folders_label = 'labels',
        pref_image='',
        pref_label='',
        ):
        super(ctechProvide, self).__init__( );
        
        self.base_folder     = base_folder
        self.sub_folders     = sub_folder
        self.folders_image   = folders_image
        self.folders_label   = folders_label
        self.data = []



    def getimage(self,i):
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;
        image_pathname = self.data[i][1];
        #load image
        image = np.array(self._loadimage(image_pathname), dtype=np.uint8)
        return image

    def getid(self): return self.data[self.index][0]
    

    def __getitem__(self, i):
                
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;

        #load data
        image_pathname = self.data[i][1]; 
        label_pathname = self.data[i][2];

        image = np.array(self._loadimage(image_pathname), dtype=np.uint8)
        label = np.array(self._loadimage(label_pathname), dtype=np.uint8)
        label = (label>128).astype( np.uint8 )
      
        return image, label
            

    def load_folders(self):
        '''
        load file patch for disk
        '''
        
        self.data = []                
        folder_path = os.path.join(self.base_folder, self.sub_folders )
        folder_image = os.path.join(folder_path, self.folders_image)
        folder_label = os.path.join(folder_path, self.folders_label)

        data =  [(f,
                os.path.join(folder_image,'{}'.format(f)),
                os.path.join(folder_label,'{}'.format(f)),
                ) for f in sorted(os.listdir( folder_image )) ]         

        self.data = data


class nucleiProvide(dataProvide):
    '''
    Mnagement for NUCLEI SEGMENTATION dataset
    https://nucleisegmentationbenchmark.weebly.com/
    '''

    @classmethod
    def create(
        cls, 
        base_folder,
        sub_folder, 
        folders_image='images',
        folders_label='labels',
        ):
        '''
        Factory function that create an instance of ctechProvide and load the data form disk.
        '''
        provide = cls(base_folder, sub_folder ,folders_image, folders_label)
        provide.load_folders();
        return provide

    
    def __init__(self,
        base_folder,    
        sub_folder,     
        folders_image = 'images',
        folders_label = 'labels',
        pref_image='',
        pref_label='',
        ):
        super(nucleiProvide, self).__init__( );
        
        self.base_folder     = base_folder
        self.sub_folders     = sub_folder
        self.folders_image   = folders_image
        self.folders_label   = folders_label
        self.data = []

    def getimage(self,i):
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;
        image_pathname = self.data[i][1];
        #load image
        image = np.array(self._loadimage(image_pathname), dtype=np.uint8)
        return image

    def getid(self): return self.data[self.index][0]
    

    def __getitem__(self, i):
                
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;

        #load data
        image_pathname = self.data[i][1]; 
        label_pathname = self.data[i][2];
        image = np.array(self._loadimage(image_pathname), dtype=np.uint8)
        label = np.array(self._loadimage(label_pathname), dtype=np.uint8)
        return image, label
            

    def load_folders(self):
        '''
        load file patch for disk
        '''
        self.data = []                
        folder_path = os.path.join(self.base_folder, self.sub_folders )
        folder_image = os.path.join(folder_path, self.folders_image)
        folder_label = os.path.join(folder_path, self.folders_label)

        data =  [(f.split('.')[0],
                os.path.join(folder_image,'{}'.format(f)),
                os.path.join(folder_label,'{}{}'.format(f.split('.')[0], '.png' )),
                ) for f in sorted(os.listdir( folder_image )) ]         

        self.data = data




class imageProvide(dataProvide):
    '''
    Management the image resources  
    '''

    def __init__(self, path, ext='jpg', fn_image=''):
        super(imageProvide, self).__init__( );
        
        if os.path.isdir(path) is not True:
            raise ValueError('Path {} is not directory'.format(path))
        
        self.fn_image = fn_image;
        self.path = path;
        self.pathimage = os.path.join(path, fn_image);

        #self.files = os.listdir(self.pathimage);
        self.data = [ f for f in sorted(os.listdir(self.pathimage)) if f.split('.')[-1] == ext ];    
        assert( len(self.data) )
        self.ext = ext;


    def __getitem__(self, idx):
        return self.getimage(idx)

    def getimage(self, i):
        '''
        Get image i
        '''
        #check index
        if i<0 and i>self.num: raise ValueError('Index outside range');
        self.index = i;

        pathname = os.path.join(self.path,self.fn_image, self.data[i]);        
        return np.array(self._loadimage(pathname));
    
    def getid(self):
        '''
        Get current image name
        '''
        return self.data[self.index];
