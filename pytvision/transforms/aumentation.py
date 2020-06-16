
import torch
import numpy as np
import cv2

from .grid.grid_sample import grid_sample
from .grid.tps_grid_gen import TPSGridGen

from . import functional as F


interpolate_image_mode   = cv2.INTER_LINEAR
interpolate_mask_mode    = cv2.INTER_NEAREST
interpolate_weight_mode  = cv2.INTER_LINEAR


class ObjectTransform(object):
    def __init__(self ):
        pass

    def size(self):
        pass

    #pytorch transform
    def to_tensor(self):
        pass

    ##interface of dict output
    def to_dict(self):
        pass

    ##interface of value/tupla output
    def to_value(self):
        pass


class ObjectRegressionTransform( ObjectTransform ):
    def __init__(self, x, y ):
        self.x = x
        self.y = y

    def size(self):
        return x.shape[0]

    #pytorch transform
    def to_tensor(self):
        x = np.array( self.x )
        y = np.array( self.y )
        self.x = torch.from_numpy( x ).float()
        self.y = torch.from_numpy( y ).float()

    ##interface of dict output
    def to_dict(self):
        return { 'x':x, 'y':y }

    ##interface of value/tupla output
    def to_value(self):
        return self.x, self.y

class ObjectImageTransform( ObjectTransform ):


    def __init__(self, image ):
        self.image = image

    def size(self): return self.image.shape

    #blur transforms

    ### lineal blur transform
    def lineal_blur(self, gen):
        self.image, _ = gen.generatelineal( self.image )

    ### motion blur transform
    def motion_blur(self, gen):
        self.image, _, _ = gen.generatecurve( self.image )

    ### gaussian blur
    def gaussian_blur(self, wnd):
        self.image = cv2.GaussianBlur(self.image, (wnd, wnd), 0)

    #colors transforms

    ### add noice
    def add_noise(self, noise):

        image = self.image
        assert( np.any( image.shape[:2] == noise.shape ) )

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        gray, a, b = cv2.split(lab)
        gray = gray.astype(np.float32)/255

        H,W  = gray.shape
        noisy = gray + noise
        noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)

        lab   = cv2.merge((noisy, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        self.image = image

    ### brightness
    def brightness(self, alpha):
        img = np.copy( self.image )
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[..., :3] = F.clip(alpha * img[...,:3].astype(np.float32), dtype, maxval)
        self.image = img

    ### brightness shift
    def brightness_shift(self, alpha, scale_value):
        img = np.copy( self.image )
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[..., :3] = F.clip(alpha * scale_value + img[...,:3].astype(np.float32), dtype, maxval)
        self.image = img

    ### contrast
    def contrast(self, alpha):
        img = np.copy( self.image )
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[:, :, :3] = F.clip(alpha * img[:, :, :3].astype(np.float32) + gray, dtype, maxval)
        self.image = img

    ### saturation
    #REVIEW!!!!
    def saturation(self, alpha):
        img = np.copy( self.image )
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype( np.float32 )
        img[..., :3] = alpha * img[..., :3].astype( np.float32 ) + (1.0 - alpha) * gray
        img[..., :3] = F.clip(img[..., :3], dtype, maxval)
        self.image = img

    ### hue saturation shift
    def hue_saturation_shift(self, alpha):
        image = np.copy( self.image )
        h   = int(alpha*180)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 170
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        self.image = image

    ### hue saturation
    def hue_saturation(self, hue_shift, sat_shift, val_shift):
        image = np.copy( self.image )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image)
        h = cv2.add(h, hue_shift)
        s = cv2.add(s, sat_shift)
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        self.image = image

    ### rgb shift
    def rgbshift(self, r_shift, g_shift, b_shift):
        image = np.copy( self.image )
        r,g,b = cv2.split(image)
        r = cv2.add(r, r_shift)
        g = cv2.add(g, g_shift)
        b = cv2.add(b, b_shift)
        image = cv2.merge((r, g, b))
        self.image = image

    ### gamma correction
    def gamma_correction(self, gamma):
        image = np.copy( self.image )
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table) # apply gamma correction using the lookup table
        self.image = image

    ### to gray
    def to_gray(self):
        image = np.copy( self.image )
        grayimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.image = cv2.cvtColor(grayimage, cv2.COLOR_GRAY2RGB)

    ### to negative
    def to_negative(self):
        image = np.copy( self.image )
        self.image = 255 - image

    ### rgb chanels permutation
    def rgbpermutation(self, indexs):
        image = np.copy( self.image )
        self.image =  image[:,:, indexs ]

    ### histogram ecualization
    def clahe(self, clipLimit, tileGridSize):
        im = np.copy( self.image )
        img_yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        self.image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    ### mean normalization
    def mean_normalization(self, mean, std):
        tensor = self.image.float()/255.0
        result_tensor = []
        for t, m, s in zip(tensor, mean, std):
            result_tensor.append(t.sub_(m).div_(s))
        self.image = torch.stack(result_tensor, 0)

    ### white normalization
    def white_normalization(self):
        tensor = self.image.float()
        new_tensor = []
        for t in tensor:
            t = t.sub_( t.min() )
            t = t.div_( t.max() )
            new_tensor.append( t )
        self.image = torch.stack(new_tensor, 0)

    ### normalization
    def normalization(self):
        self.image = self.image.float()/255.0

    ### equalization
    def eq_normalization(self, A, A_pinv):
        self.image = F.equalization( self.image, A, A_pinv  )

    #Geometric transforms

    def crop( self, box, padding_mode ):
        """Crop: return if validate crop
        """
        self.image = F.imcrop( self.image, box, padding_mode )
        return True

    def scale( self, factor, padding_mode = cv2.BORDER_CONSTANT ):
        self.image = F.scale( self.image, factor, interpolate_image_mode, padding_mode )

    def pad( self, h_pad = 2, w_pad = 2, padding_mode = cv2.BORDER_CONSTANT ):
        self.image = F.pad(self.image, h_pad, w_pad, padding_mode)

    def hflip(self):
        self.image = F.hflip( self.image )

    def vflip(self):
        self.image = F.vflip( self.image )

    def rotate90(self):
        self.image = F.rotate90( self.image )

    def rotate180(self):
        self.image = F.rotate180( self.image )

    def rotate270(self):
        self.image = F.rotate270( self.image )

    def applay_geometrical_transform(self, mat_r, mat_t, mat_w, padding_mode = cv2.BORDER_CONSTANT ):
        self.image = F.applay_geometrical_transform( self.image, mat_r, mat_t, mat_w, interpolate_image_mode, padding_mode )
        return True

    def applay_elastic_transform(self, mapx, mapy, padding_mode = cv2.BORDER_CONSTANT):
        self.image  = cv2.remap(self.image,  mapx, mapy, interpolate_image_mode, borderMode=padding_mode)

    def applay_elastic_tensor_transform(self, grid):
        tensor = torch.unsqueeze( self.image, dim=0 )
        self.image = grid_sample(tensor, grid ).data[0,...]

    ### resize
    def resize(self, imsize, resize_mode, padding_mode):
        self.image = F.resize_image(self.image, imsize[1], imsize[0], resize_mode, padding_mode, interpolate_mode=interpolate_image_mode   )

    ### resize unet input
    def resize_unet_input( self, fov_size=388, padding_mode = cv2.BORDER_CONSTANT ):
        self.image = F.resize_unet_transform(self.image, fov_size, interpolate_image_mode,  padding_mode)

    #pytorch transform
    def to_tensor(self):
        image  = self.image
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        self.image = image

    ##interface of dict output
    def to_dict(self):
        return { 'image': self.image }

    ##interface of value/tupla output
    def to_value(self):
        return self.image

    # Aux function for debug
    def _draw_grid(self, grid_size=50, color=(255,0,0), thickness=1):
        image = np.copy( self.image )
        self.image = F.draw_grid(image, grid_size, color, thickness)



class ObjectImageMetadataTransform( ObjectImageTransform ):
    def __init__(self, image, meta ):
        """
        Arg:
            @image
            @meta
        """
        super(ObjectImageMetadataTransform, self).__init__(image)
        self.meta = meta


    #pytorch transform
    def to_tensor(self):

        image = self.image
        meta  = self.meta

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image  = image.transpose((2, 0, 1))
        image  = torch.from_numpy(image).float()
        meta   = torch.from_numpy( meta ).float()

        self.image  = image
        self.meta   = meta

    ##interface of output
    def to_dict(self):
        return {
            'image': self.image,
            'metadata': self.meta,
             }

    def to_value(self):
        return self.image, self.meta



class ObjectImageAndAnnotations( ObjectImageTransform ):
    def __init__(self, image, annotations, labels ):
        """
        Arg:
            @image
            @annotations
            @labels
        """
        super(ObjectImageAndAnnotations, self).__init__(image)
        self.annotations = annotations
        self.labels = labels

    ### resize
    def resize(self, imsize, resize_mode, padding_mode):

        self.image , self.annotations = F.resize_image_box(
            self.image,
            self.annotations,
            imsize[1], imsize[0],
            resize_mode,
            padding_mode,
            interpolate_mode=interpolate_image_mode
            )


    #pytorch transform
    def to_tensor(self):

        image        = self.image
        annotations  = self.annotations
        labels       = self.labels

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image       = image.transpose((2, 0, 1))
        image       = torch.from_numpy(image).float()
        annotations = torch.from_numpy( annotations ).float()
        labels      = torch.from_numpy( labels ).float()

        self.image       = image
        self.annotations = annotations
        self.labels      = labels

    ##interface of output
    def to_dict(self):
        return {
            'image': self.image,
            'annotations': self.annotations,
            'labels': self.labels
             }

    def to_value(self):
        return self.image, self.annotations, labels

class ObjectImageAndLabelTransform( ObjectImageTransform ):
    def __init__(self, image, label ):
        """
        Arg:
            @image
            @label
        """
        super(ObjectImageAndLabelTransform, self).__init__(image)
        self.label = label

    #pytorch transform
    def to_tensor(self):

        image  = self.image
        label  = self.label

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        self.image = image
        self.label = label

    ##interface of output
    def to_dict(self):
        return {
            'image': self.image,
            'label': self.label
             }

    def to_value(self):
        return self.image, self.label

class ObjectImageAndMaskTransform( ObjectImageTransform ):
    def __init__(self, image, mask):
        """
        Arg:
            @image
            @mask
        """
        super(ObjectImageAndMaskTransform, self).__init__(image)
        self.mask = mask


   #Geometric transforms
    def crop( self, box, padding_mode):
        """Crop: return if validate crop
        """

        image = F.imcrop( self.image, box, padding_mode )
        mask = F.imcrop( self.mask, box, padding_mode )

        if mask.sum() > 10: #area>10
             self.image = image
             self.mask = mask
             return True

        return False

    def scale( self, factor, padding_mode = cv2.BORDER_CONSTANT ):
        self.image = F.scale( self.image, factor, interpolate_image_mode, padding_mode )
        self.mask  = F.scale( self.mask,  factor, interpolate_mask_mode,  padding_mode )

    def pad( self, h_pad = 2, w_pad = 2, padding_mode = cv2.BORDER_CONSTANT ):
        self.image = F.pad(self.image, h_pad, w_pad, padding_mode)
        self.mask  = F.pad(self.mask,  h_pad, w_pad, padding_mode)

    def hflip(self):
        self.image = F.hflip( self.image )
        self.mask  = F.hflip( self.mask )

    def vflip(self):
        self.image = F.vflip( self.image )
        self.mask  = F.vflip( self.mask )

    def rotate90(self):
        self.image = F.rotate90( self.image )
        self.mask  = F.rotate90( self.mask )

    def rotate180(self):
        self.image = F.rotate180( self.image )
        self.mask  = F.rotate180( self.mask )

    def rotate270(self):
        self.image = F.rotate270( self.image )
        self.mask  = F.rotate270( self.mask )

    def applay_geometrical_transform(self, mat_r, mat_t, mat_w, padding_mode = cv2.BORDER_CONSTANT ):
        self.image = F.applay_geometrical_transform( self.image, mat_r, mat_t, mat_w, interpolate_image_mode, padding_mode )
        self.mask  = F.applay_geometrical_transform( self.mask,  mat_r, mat_t, mat_w, interpolate_mask_mode,  padding_mode )
        return True

    def applay_elastic_transform(self, mapx, mapy, padding_mode = cv2.BORDER_CONSTANT):
        self.image = cv2.remap(self.image,  mapx, mapy, interpolate_image_mode, borderMode=padding_mode)
        self.mask  = cv2.remap(self.mask,   mapx, mapy, interpolate_mask_mode,  borderMode=padding_mode)

    def applay_elastic_tensor_transform(self, grid):
        self.image = grid_sample( torch.unsqueeze( self.image, dim=0 ), grid ).data[0,...]
        self.mask  = grid_sample( torch.unsqueeze( self.mask, dim=0 ), grid ).round().data[0,...]

    #pytorch transform
    def to_tensor(self):

        image  = self.image
        mask   = self.mask
        mask = (mask>0).astype( np.uint8 )

        # numpy image: H x W x C
        # torch image: C X H X W
        image  = image.transpose((2, 0, 1)).astype(np.float)
        mask   = mask.transpose((2, 0, 1)).astype(np.float)
        self.image = torch.from_numpy(image).float()
        self.mask  = torch.from_numpy(mask).float()

    ### resize
    def resize(self, imsize, resize_mode, padding_mode):
        self.image = F.resize_image(self.image, imsize[1], imsize[0],  resize_mode, padding_mode, interpolate_mode=interpolate_image_mode )
        self.mask  = F.resize_image(self.mask, imsize[1], imsize[0],  resize_mode, padding_mode, interpolate_mode=interpolate_mask_mode )

    #geometric transformation
    def resize_unet_input( self, fov_size=388, padding_mode = cv2.BORDER_CONSTANT ):
        self.image = F.resize_unet_transform(self.image, fov_size, interpolate_image_mode, padding_mode)
        self.mask  = F.resize_unet_transform(self.mask , fov_size, interpolate_mask_mode,  padding_mode)

    ##interface of output
    def to_dict(self):
        return {
            'image': self.image,
            'label': self.mask
             }

    def to_value(self):
        return self.image, self.mask

class ObjectImageAndMaskMetadataTransform( ObjectImageAndMaskTransform ):
    def __init__(self, image, mask, metadata):
        """
        Arg:
            @image
            @mask
            @metadata
        """
        super(ObjectImageAndMaskMetadataTransform, self).__init__(image, mask)
        self.metadata = metadata

    #pytorch transform
    def to_tensor(self):

        image  = self.image
        mask   = self.mask
        meta   = self.metadata
        mask = (mask>0).astype( np.uint8 )

        # numpy image: H x W x C
        # torch image: C X H X W
        image  = image.transpose((2, 0, 1)).astype(np.float)
        mask   = mask.transpose((2, 0, 1)).astype(np.float)
        self.image = torch.from_numpy(image).float()
        self.mask  = torch.from_numpy(mask).float()
        self.metadata = torch.from_numpy(meta).float()

    ##interface of output
    def to_dict(self):
        return {
            'image': self.image,
            'label': self.mask,
            'metadata': self.metadata
             }

    def to_value(self):
        return self.image, self.mask, self.metadata

class ObjectImageMaskAndWeightTransform(ObjectImageAndMaskTransform):
    def __init__(self, image, mask, weight ):
        """
        Arg:
            @image
            @mask
            @weight
        """
        super(ObjectImageMaskAndWeightTransform, self).__init__(image, mask)
        self.weight = weight


    #pytorch transform
    def to_tensor(self):

        image  = self.image
        mask   = self.mask
        weight = self.weight
        mask = (mask>0).astype( np.uint8 )

        # numpy image: H x W x C
        # torch image: C X H X W
        image  = image.transpose((2, 0, 1)).astype(np.float)
        mask   = mask.transpose((2, 0, 1)).astype(np.float)
        weight = weight.transpose((2, 0, 1)).astype(np.float)

        self.image  = torch.from_numpy(image).float()
        self.mask   = torch.from_numpy(mask).float()
        self.weight = torch.from_numpy(weight).float()


    #Geometric transformation

    def crop( self, box, padding_mode):
        """Crop: return if validate crop
        """
        image = F.imcrop( self.image, box, padding_mode )
        mask = F.imcrop( self.mask, box, padding_mode )
        weight = F.imcrop( self.weight, box, padding_mode )

        if mask.sum() > 10: #area>10
            self.image = image
            self.mask = mask
            self.weight = weight
            return True

        return False

    def scale( self, factor, padding_mode = cv2.BORDER_CONSTANT ):
        self.image  = F.scale( self.image,  factor, interpolate_image_mode,  padding_mode )
        self.mask   = F.scale( self.mask,   factor, interpolate_mask_mode,   padding_mode )
        self.weight = F.scale( self.weight, factor, interpolate_weight_mode, padding_mode )

    def pad( self, h_pad = 2, w_pad = 2, padding_mode = cv2.BORDER_CONSTANT ):
        self.image  = F.pad(self.image,  h_pad, w_pad, padding_mode)
        self.mask   = F.pad(self.mask,   h_pad, w_pad, padding_mode)
        self.weight = F.pad(self.weight, h_pad, w_pad, padding_mode)

    def hflip(self):
        self.image  = F.hflip( self.image )
        self.mask   = F.hflip( self.mask )
        self.weight = F.hflip( self.weight )

    def vflip(self):
        self.image  = F.vflip( self.image )
        self.mask   = F.vflip( self.mask )
        self.weight = F.vflip( self.weight )

    def rotate90(self):
        self.image  = F.rotate90( self.image )
        self.mask   = F.rotate90( self.mask )
        self.weight = F.rotate90( self.weight )

    def rotate180(self):
        self.image  = F.rotate180( self.image )
        self.mask   = F.rotate180( self.mask )
        self.weight = F.rotate180( self.weight )

    def rotate270(self):
        self.image  = F.rotate270( self.image )
        self.mask   = F.rotate270( self.mask )
        self.weight = F.rotate270( self.weight )

    def applay_geometrical_transform(self, mat_r, mat_t, mat_w, padding_mode = cv2.BORDER_CONSTANT ):
        self.image  = F.applay_geometrical_transform( self.image,  mat_r, mat_t, mat_w, interpolate_image_mode,  padding_mode )
        self.mask   = F.applay_geometrical_transform( self.mask,   mat_r, mat_t, mat_w, interpolate_mask_mode,   padding_mode )
        self.weight = F.applay_geometrical_transform( self.weight, mat_r, mat_t, mat_w, interpolate_weight_mode, padding_mode )
        return True

    def applay_elastic_transform(self, mapx, mapy, padding_mode = cv2.BORDER_CONSTANT):
        self.image  = F.cunsqueeze( cv2.remap(self.image,  mapx, mapy, interpolate_image_mode,  borderMode=padding_mode) )
        self.mask   = F.cunsqueeze( cv2.remap(self.mask,   mapx, mapy, interpolate_mask_mode,   borderMode=padding_mode) )
        self.weight = F.cunsqueeze( cv2.remap(self.weight, mapx, mapy, interpolate_weight_mode, borderMode=padding_mode) )

    def applay_elastic_tensor_transform(self, grid):
        self.image  = grid_sample( torch.unsqueeze( self.image,  dim=0 ), grid ).data[0,...]
        self.mask   = grid_sample( torch.unsqueeze( self.mask,   dim=0 ), grid ).round().data[0,...]
        self.weight = grid_sample( torch.unsqueeze( self.weight, dim=0 ), grid ).data[0,...]

    ### resize
    def resize(self, imsize, resize_mode, padding_mode):
        self.image  = F.resize_image(self.image,  imsize[1], imsize[0],  resize_mode, padding_mode, interpolate_mode=interpolate_image_mode  )
        self.mask   = F.resize_image(self.mask,   imsize[1], imsize[0],  resize_mode, padding_mode, interpolate_mode=interpolate_mask_mode   )
        self.weight = F.resize_image(self.weight, imsize[1], imsize[0],  resize_mode, padding_mode, interpolate_mode=interpolate_weight_mode )

    def resize_unet_input( self, fov_size=388, padding_mode = cv2.BORDER_CONSTANT ):
        super(ObjectImageMaskAndWeightTransform, self).resize_unet_input(fov_size, padding_mode)
        self.weight = F.resize_unet_transform(self.weight, fov_size,  interpolate_weight_mode,  padding_mode)

    ##interface of output
    def to_dict(self):
        return {
            'image': self.image,
            'label': self.mask,
            'weight': self.weight,
             }

    def to_value(self):
        return self.image, self.mask, self.weight

class ObjectImageMaskMetadataAndWeightTransform( ObjectImageMaskAndWeightTransform ):
    def __init__(self, image, mask, weight, metadata):
        """
        Arg:
            @image
            @mask
            @weight
            @metadata
        """
        super(ObjectImageMaskMetadataAndWeightTransform, self).__init__(image, mask, weight)
        self.metadata = metadata

    #pytorch transform
    def to_tensor(self):

        image  = self.image
        mask   = self.mask
        weight = self.weight
        meta   = self.metadata
        mask = (mask>0).astype( np.uint8 )

        # numpy image: H x W x C
        # torch image: C X H X W
        image  = image.transpose((2, 0, 1)).astype(np.float)
        mask   = mask.transpose((2, 0, 1)).astype(np.float)
        weight = weight.transpose((2, 0, 1)).astype(np.float)

        self.image = torch.from_numpy(image).float()
        self.mask  = torch.from_numpy(mask).float()
        self.weight = torch.from_numpy(weight).float()
        self.metadata = torch.from_numpy(meta).float()

    ##interface of output
    def to_dict(self):
        return {
            'image': self.image,
            'label': self.mask,
            'weight': self.weight,
            'metadata': self.metadata
             }

    def to_value(self):
        return self.image, self.mask, self.weight, self.metadata



class ObjectImage2ImageTransform(ObjectTransform):
    def __init__(self, x, y ):
        self.x = x
        self.y = y

    def size(self):
        return [self.x.shape[1],self.x.shape[2],self.x.shape[0]]


    #blur transforms

    ### lineal blur transform
    def lineal_blur(self, gen):
        xnew = []
        for i,e in enumerate(self.x):
            e, _ = gen.generatelineal( F.cunsqueeze(e) )
            xnew.append(e)
        self.x = np.stack(xnew,axis=0)

    ### motion blur transform
    def motion_blur(self, gen):
        xnew = []
        for i,e in enumerate(self.x):
            e, _ = gen.generatecurve( F.cunsqueeze(e) )
            xnew.append(e[:,:,0])
        self.x = np.stack(xnew,axis=0)

    ### gaussian blur
    def gaussian_blur(self, wnd):
        xnew = []
        for i,e in enumerate(self.x):
            e, _ = cv2.GaussianBlur( F.cunsqueeze(e), (wnd, wnd), 0 )
            xnew.append(e[:,:,0])
        self.x = np.stack(xnew,axis=0)


    #Geometric transformation

    def crop( self, box, padding_mode):
        """Crop: return if validate crop
        """
        xnew = []
        for i,e in enumerate(self.x):
            xnew.append(F.imcrop(F.cunsqueeze(e), box, padding_mode )[:,:,0])
        self.x = np.stack(xnew,axis=0)
        self.y = F.imcrop( self.y, box, padding_mode )
        return True

    def scale( self, factor, padding_mode = cv2.BORDER_CONSTANT ):
        xnew = []
        for i,e in enumerate(self.x):
            xnew.append(F.scale(F.cunsqueeze(e), factor, interpolate_image_mode, padding_mode )[:,:,0])
        self.x = np.stack(xnew,axis=0)
        self.y = F.scale( self.y, factor, interpolate_image_mode, padding_mode )

    def pad( self, h_pad = 2, w_pad = 2, padding_mode = cv2.BORDER_CONSTANT ):
        xnew = []
        for i,e in enumerate(self.x):
            xnew.append(F.pad(F.cunsqueeze(e), h_pad, w_pad, padding_mode )[:,:,0])
        self.x = np.stack(xnew, axis=0)
        self.y = F.pad(self.y, h_pad, w_pad, padding_mode )

    def hflip(self):
        xnew = []
        for i,e in enumerate(self.x):
            xnew.append(F.hflip(F.cunsqueeze(e))[:,:,0])
        self.x = np.stack(xnew, axis=0)
        self.y = F.hflip(self.y)

    def vflip(self):
        xnew = []
        for i,e in enumerate(self.x):
            xnew.append(F.vflip(F.cunsqueeze(e))[:,:,0])
        self.x = np.stack(xnew, axis=0)
        self.y = F.vflip(self.y)

    def rotate90(self):
        xnew = []
        for i,e in enumerate(self.x):
            xnew.append(F.rotate90(F.cunsqueeze(e))[:,:,0])
        self.x = np.stack(xnew, axis=0)
        self.y = F.rotate90(self.y)

    def rotate180(self):
        xnew = []
        for i,e in enumerate(self.x):
            xnew.append(F.rotate180(F.cunsqueeze(e))[:,:,0])
        self.x = np.stack(xnew, axis=0)
        self.y = F.rotate180(self.y)

    def rotate270(self):
        xnew = []
        for i,e in enumerate(self.x):
            xnew.append(F.rotate270(F.cunsqueeze(e))[:,:,0])
        self.x = np.stack(xnew, axis=0)
        self.y = F.rotate270(self.y)

    def resize(self, imsize, resize_mode, padding_mode):
        xnew = []
        for i,e in enumerate(self.x):
            xnew.append(F.resize_image(F.cunsqueeze(e), imsize[1], imsize[0], resize_mode, padding_mode, interpolate_mode=interpolate_image_mode)[:,:,0])
        self.x = np.stack(xnew, axis=0)
        self.y = F.resize_image(self.y, imsize[1], imsize[0], resize_mode, padding_mode, interpolate_mode=interpolate_image_mode)

    def resize_unet_input( self, fov_size=388, padding_mode = cv2.BORDER_CONSTANT ):
        xnew = []
        for i,e in enumerate(self.x):
            xnew.append(F.resize_unet_transform(F.cunsqueeze(e), fov_size, interpolate_image_mode, padding_mode)[:,:,0])
        self.x = np.stack(xnew, axis=0)
        self.y = F.resize_unet_transform(self.y, fov_size, interpolate_image_mode, padding_mode)


    #normalization

    ### mean
    def mean_normalization(self, mean, std):
        tensor = self.x
        result_tensor = []
        for t, m, s in zip(tensor, mean, std):
            result_tensor.append(t.sub_(m).div_(s))
        self.x = torch.stack(result_tensor, 0)

    ### white
    def white_normalization(self):
        tensor = self.x.float()
        new_tensor = []
        for t in tensor:
            t = t.sub_( t.min() )
            t = t.div_( t.max() )
            new_tensor.append( t )
        self.x = torch.stack(new_tensor, 0)

    #pytorch transform
    def to_tensor(self):
        x = np.array( self.x )
        y = np.array( self.y ).transpose((2, 0, 1)).astype(np.float)

        self.x = torch.from_numpy( x ).float()
        self.y = torch.from_numpy( y ).float()

    ##interface of dict output
    def to_dict(self):
        return { 'x':self.x, 'y':self.y }

    ##interface of value/tupla output
    def to_value(self):
        return self.x, self.y