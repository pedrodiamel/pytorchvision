
import random
import numpy as np
import cv2
import types

from .renderblur import BlurRender
from .aumentation import ObjectTransform
from . import functional as F



class ToTransform(object):
    """Abstrat class of Generic transform 
    """
    
    def __init__(self):
        pass        
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
        
    
    def __str__(self):
        return self.__class__.__name__

class ToRandomTransform(ToTransform):
    """Random transform: 
    """
    
    def __init__(self, tran, prob):
        """Initialization
        Args:
            @tran: class tranform 
            @prob: probability
        """
        self.tran = tran 
        self.prob=prob
        
        
    def __call__(self,obj):
        if random.random() < self.prob:
            obj = self.tran( obj )
        return obj

class ToRandomChoiceTransform(ToTransform):
    """Random choice transform: 
    """
    
    def __init__(self, transforms):
        """Initialization
        Args:
            @transforms: list of the transforms  
        """
        assert(len(transforms))
        self.transforms=transforms
        
        
    def __call__(self,obj):
        tran = random.choice( self.transforms )
        obj = tran( obj )
        return obj

class ToRandomOrderTransform(ToTransform):
    """Random order transform: 
    """
    
    def __init__(self, transforms):
        """Initialization
        Args:
            @transforms: list of the transforms  
        """
        assert(len(transforms))
        self.transforms=transforms
        
        
    def __call__(self,obj):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            obj = self.transforms[i](obj)
        return obj
    
class ToLambdaTransform(ToTransform):
    """Random lambda transform: 
    """
    
    def __init__(self, lambd):
        """Initialization
        Args:
            @lambd (function): Lambda/function to be used for transform.
        """
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambds
        
    def __call__(self,obj):
        return self.lambd(obj)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, obj):
        obj.to_tensor()
        return obj

# Blur transformation

class ToLinealMotionBlur(ToTransform):
    """Lineal Blur randomly.
    """

    def __init__(self, lmax=100  ):        
        """Initialization
        Args:
            @lmax: maximun lineal blur
        """
        self.gen = BlurRender(lmax)

    def __call__(self, obj):
        obj.lineal_blur(self.gen)
        return obj

class ToMotionBlur(ToTransform):
    """Motion Blur randomly.
    """

    def __init__(self,
        pSFsize=64,
        maxTotalLength=64,
        anxiety=0.005,
        numT=2000,
        texp=0.75, 
        ):        
        """Initialization
        Args:
            @pSFsize: kernel size (psf)
            @maxTotalLength: length trayectory
            @anxiety:
            @numT:
            @texp:
        """
        self.gen = BlurRender(pSFsize, maxTotalLength, anxiety, numT, texp)

    def __call__(self, obj):
        obj.motion_blur(self.gen)
        return obj

class ToGaussianBlur(ToTransform):
    """Gaussian Blur randomly.
    """

    def __init__(self, sigma=0.2 ):        
        """Initialization
        Args:
            @lmax: maximun lineal blur
        """
        self.sigma = sigma

    def __call__(self, obj):
        
        # add gaussian noise
        H,W = obj.size()[:2]
        noise = np.array([random.gauss(0,self.sigma) for i in range(H*W)])
        noise = noise.reshape(H,W)
        obj.add_noise( noise )
        wnd = random.randint(1,3) * 2 + 1
        obj.gaussian_blur(wnd)
        return obj


class ToGaussianNoise(ToTransform):
    """Gaussian Noise randomly.
    """
    def __init__(self, sigma=0.2 ):        
        """Initialization
        Args:
            @lmax: maximun lineal blur
        """
        self.sigma = sigma

    def __call__(self, obj):
        
        # add gaussian noise
        H,W = obj.size()[:2]
        noise = np.array([random.gauss(0,self.sigma) for i in range(H*W)])
        noise = noise.reshape(H,W)
        obj.add_noise( noise )
        return obj

# Color tranformations

class RandomBrightness(ToTransform):
    """Random Brightness.
    """
    def __init__(self, factor=0.5 ):        
        """Initialization
        Args:
            @factor: factor
        """
        self.factor = factor

    def __call__(self, obj):
        alpha = 1.0 + self.factor*random.uniform(-1, 1)
        obj.brightness(alpha)
        return obj

class RandomBrightnessShift(ToTransform):
    """Random Brightness Shift.
    """
    def __init__(self, factor=0.125, scale_value=100 ):        
        """Initialization
        Args:
            @factor: factor
        """
        self.factor = factor
        self.scale_value = scale_value

    def __call__(self, obj):
        alpha = 1.0 + (self.factor)*random.uniform(-1, 1)
        obj.brightness_shift(alpha, self.scale_value)
        return obj

class RandomContrast(ToTransform):
    """Random Contrast.
    """
    def __init__(self, factor=0.3 ):        
        """Initialization
        Args:
            @factor: factor
        """
        self.factor = factor

    def __call__(self, obj):
        alpha = 1.0 + self.factor*random.uniform(-1, 1)
        obj.contrast(alpha)
        return obj

class RandomSaturation(ToTransform):
    """Random Saturation.
    """
    def __init__(self, factor=0.75 ):        
        """Initialization
        Args:
            @factor: factor
        """
        self.factor = factor

    def __call__(self, obj):
        alpha = 1.0 + self.factor*random.uniform(-1, 1)
        obj.saturation(alpha)
        return obj

class RandomHueSaturationShift(ToTransform):
    """Random Hue Saturation Shift.
    """
    def __init__(self, factor=0.3 ):        
        """Initialization
        Args:
            @factor: factor
        """
        self.factor = factor

    def __call__(self, obj):
        alpha = 1.0 + random.uniform(-self.factor, self.factor)
        obj.hue_saturation_shift(alpha)
        return obj

class RandomHueSaturation(ToTransform):
    """Random Hue Saturation.
    """
    def __init__(self, hue_shift_limit=(-5, 5), sat_shift_limit=(-11, 11), val_shift_limit=(-11, 11)):        
        """Initialization
        Args:
            @hue_shift_limit: hue_shift_limit
            @sat_shift_limit: sat_shift_limit
            @val_shift_limit: val_shift_limit
        """
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit

    def __call__(self, obj):        
        hue_shift = random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
        sat_shift = random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
        val_shift = random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
        obj.hue_saturation(hue_shift, sat_shift, val_shift)
        return obj

class RandomRGBShift(ToTransform):
    """Random RGB Shift.
    """
    def __init__(self, r_shift_limit=(-128, 128), g_shift_limit=(-128, 128), b_shift_limit=(-128, 128)):        
        """Initialization
        Args:
            @r_shift_limit: r_shift_limit
            @g_shift_limit: g_shift_limit
            @b_shift_limit: b_shift_limit
        """
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit

    def __call__(self, obj):        
        r_shift = random.uniform(self.r_shift_limit[0], self.r_shift_limit[1])
        g_shift = random.uniform(self.g_shift_limit[0], self.g_shift_limit[1])
        b_shift = random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])
        obj.rgbshift(r_shift, g_shift, b_shift)
        return obj

class RandomGamma(ToTransform):
    """Random Gamma.
    """
    def __init__(self, factor=0.5 ):        
        """Initialization
        Args:
            @factor: factor
        """
        self.factor = factor

    def __call__(self, obj):
        alpha = 1.0 + self.factor*random.uniform(-1, 1)
        obj.gamma_correction(alpha)
        return obj

class ToGrayscale(ToTransform):
    """To gray image
    """

    def __init__(self):
        """Initialization
        """
        pass
    def __call__(self, obj):
        obj.to_gray()
        return obj

class ToNegative(ToTransform):
    """To negative 
    """

    def __init__(self):
        """Initialization
        """
        pass
    def __call__(self, obj):
        obj.to_negative()
        return obj

class RandomRGBPermutation(ToTransform):
    """RGB permutation 
    """
    def __init__(self):
        """Initialization
        """
        self.indexs = [0,1,2]
    def __call__(self, obj):
        random.shuffle(self.indexs)
        obj.rgbpermutation( self.indexs )
        return obj

class CLAHE(ToTransform):
    """CLAHE ecualization.
    """
    def __init__(self, clipfactor=2.0, tileGridSize=(8, 8) ):        
        """Initialization
        Args:
            @factor: factor
        """
        self.clipfactor = clipfactor
        self.tileGridSize = tileGridSize

    def __call__(self, obj):
        obj.clahe( self.clipfactor,  self.tileGridSize )
        return obj

class ToMeanNormalization(ToTransform):
    """To mean normalization 
    """

    def __init__(self, mean, std):
        """Initialization
        Args:
            @mean: mean
            @std: estandar desviation
        """
        self.mean = mean 
        self.std = std

    def __call__(self, obj):
        obj.mean_normalization(self.mean, self.std)
        return obj

class ToWhiteNormalization(ToTransform):
    """To white normalization 
    """
    def __init__(self ):
        """Initialization
        """
        pass
    def __call__(self, obj):
        obj.white_normalization()
        return obj

class ToNormalization(ToTransform):
    """To white normalization 
    """
    def __init__(self ):
        """Initialization
        """
        pass
    def __call__(self, obj):
        obj.normalization()
        return obj

class ToEqNormalization(ToTransform):
    """To equalization normalization 
    """
    def __init__(self, imsize ):
        """Initialization
        """
        self.A, self.A_pinv = F.compute_norm_mat( imsize[0], imsize[1] )

    def __call__(self, obj):
        obj.eq_normalization(self.A, self.A_pinv)
        return obj

# geometrical transforms

class ToResize(ToTransform):
    """Resize
    """
    
    def __init__(self, imsize, resize_mode=None, padding_mode=cv2.BORDER_CONSTANT ):
        """Initialization
        Args:
            @imsize: size input layer resize (w,h)
            @resize_mode: resize mode
        """
        self.imsize = imsize
        self.resize_mode = resize_mode
        self.padding_mode = padding_mode

    def __call__(self, obj):
        obj.resize( self.imsize, self.resize_mode, self.padding_mode )
        return obj

class ToPad(ToTransform):
    r"""Pad
    Args:
        h_pad: height padding
        w_pad: width  padding
        pad_mode: resize mode
    """
    
    def __init__(self, h_pad, w_pad, padding_mode=cv2.BORDER_CONSTANT ):
        self.h_pad = h_pad
        self.w_pad = w_pad
        self.padding_mode = padding_mode

    def __call__(self, obj):
        obj.pad( self.h_pad, self.w_pad, self.padding_mode )
        return obj



class ToResizeUNetFoV(ToTransform):
    """Resize to unet fov
    """
    
    def __init__(self, fov=388, padding_mode=cv2.BORDER_CONSTANT):
        """Initialization
        Args:
            @fov: size input layer for unet model
        """
        self.fov=fov
        self.padding_mode = padding_mode
        
    def __call__(self, obj):
        obj.resize_unet_input( self.fov, self.padding_mode )
        return obj

class CenterCrop(ToTransform):
    """Center Crop
    """
    
    def __init__(self, cropsize, padding_mode=cv2.BORDER_CONSTANT ):
        """Initialization
        Args:
            @cropsize [w,h]
        """
        self.cropsize = cropsize
        self.padding_mode = padding_mode
        
    def __call__(self, obj):
        h, w = obj.size()[:2]
        x = (w - self.cropsize[0]) // 2
        y = (h - self.cropsize[1]) // 2
        obj.crop( [ x, y, self.cropsize[0], self.cropsize[1] ], self.padding_mode )
        return obj

class RandomCrop(ToTransform):
    """Random Crop
    """
    
    def __init__(self, cropsize, limit=10, padding_mode=cv2.BORDER_CONSTANT ):
        """Initialization
        Args:
            @cropsize [w,h]
            @limit
        """
        self.cropsize = cropsize
        self.limit = limit
        self.padding_mode = padding_mode
        self.centecrop = CenterCrop(cropsize, padding_mode)
        
    def __call__(self, obj):
        h, w = obj.size()[:2]
        newW, newH = self.cropsize

        assert(w - newW + self.limit > 0)
        assert(h - newH + self.limit > 0)

        for _ in range(10):       
            x = random.randint( -self.limit, (w - newW) + self.limit )
            y = random.randint( -self.limit, (h - newH) + self.limit )
            box = [ x, y, self.cropsize[0], self.cropsize[1] ]
            if obj.crop( box, self.padding_mode ):
                return obj

        return self.centecrop(obj)

class RandomScale(ToTransform):
    """ SRandom Scale.
    """
    def __init__(self, factor=0.1, padding_mode=cv2.BORDER_CONSTANT,  
        ):        
        """Initialization
        Args:
            @factor: factor of scale
            @padding_mode        
        """
        self.factor = factor
        self.padding_mode = padding_mode

    def __call__(self, obj):        
        height, width = obj.size()[:2]
        factor =  1.0 + self.factor*random.uniform(-1.0, 1.0)
        obj.scale( factor, self.padding_mode )
        return obj

class HFlip(ToTransform):
    """ Horizontal Flip.
    """
    def __init__(self):        
        """Initialization 
        """
        pass

    def __call__(self, obj):
        obj.hflip()
        return obj

class VFlip(ToTransform):
    """ Vertical Flip.
    """
    def __init__(self):        
        """Initialization 
        """
        pass

    def __call__(self, obj):
        obj.vflip()
        return obj

class Rotate90(ToTransform):
    """ Rotate 90.
    """
    def __init__(self):        
        """Initialization 
        """
        pass

    def __call__(self, obj):
        obj.rotate90()
        return obj
    
class Rotate180(ToTransform):
    """ Rotate 180 .
    """
    def __init__(self):        
        """Initialization 
        """
        pass

    def __call__(self, obj):
        obj.rotate180()
        return obj

class Rotate270(ToTransform):
    """ Rotate 270 .
    """
    def __init__(self):        
        """Initialization 
        """
        pass

    def __call__(self, obj):
        obj.rotate270()
        return obj

class RandomGeometricalTransform(ToTransform):
    """ Random Geometrical Transform
    """
    def __init__(self, angle=360, translation=0.2, warp=0.0, padding_mode=cv2.BORDER_CONSTANT ):        
        """Initialization 
        Args:
            @angle: ratate angle
            @translate 
            @warp
        """
        self.angle = angle
        self.translation = translation
        self.warp = warp
        self.padding_mode = padding_mode

    def __call__(self, obj):
        
        imsize = obj.size()[:2]
        for _ in range(10):       
            mat_r, mat_t, mat_w = F.get_geometric_random_transform( imsize, self.angle, self.translation, self.warp )
            if obj.applay_geometrical_transform( mat_r, mat_t, mat_w, self.padding_mode ):
                return obj
        return obj

class RandomElasticDistort(ToTransform):
    """ Random Elastic Distort
    """

    def __init__(self, size_grid=50, deform=15, padding_mode=cv2.BORDER_CONSTANT  ):        
        """Initialization 
        Args:
            @size_grid: ratate angle
            @deform 
        """
        self.size_grid = size_grid
        self.deform = deform
        self.padding_mode = padding_mode

    def __call__(self, obj):        
        imsize = obj.size()[:2]
        mapx, mapy = F.get_elastic_transform(imsize, self.size_grid, self.deform )
        obj.applay_elastic_transform( mapx, mapy, self.padding_mode )
        return obj
        
class RandomElasticTensorDistort(object):
    """ Random Elastic Tensor Distort
    """

    def __init__(self, size_grid=10, deform=0.05  ):    
        """Initialization 
        Args:
            @size_grid: ratate angle
            @deform 
        """
        self.size_grid = size_grid
        self.deform = deform

    def __call__(self, obj):        
        width, height = obj.image.size(1), obj.image.size(2)           
        grid = F.get_tensor_elastic_transform( (height, width), self.size_grid, self.deform )
        obj.applay_elastic_tensor_transform( grid ) 
        return obj
