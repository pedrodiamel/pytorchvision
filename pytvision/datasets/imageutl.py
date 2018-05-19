
import os
import numpy as np
import PIL.Image

from . import utility as utl


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
