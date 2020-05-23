

import os
import sys
import numpy as np
import PIL.Image

import math
import cv2
import random

from scipy.interpolate import griddata

from scipy import ndimage
import scipy.misc


import torch
from torch.autograd import Variable

import itertools

from .grid.grid_sample import grid_sample
from .grid.tps_grid_gen import TPSGridGen
from .rectutils import Rect


def cunsqueeze(data):
    if len( data.shape ) == 2:
        data = data[:,:,np.newaxis]
    return data

def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)

def relabel( mask ):
    h, w = mask.shape
    relabel_dict = {}
    for i, k in enumerate(np.unique(mask)):
        if k == 0:
            relabel_dict[k] = 0
        else:
            relabel_dict[k] = i
    for i, j in product(range(h), range(w)):
        mask[i, j] = relabel_dict[mask[i, j]]
    return mask

def scale(image, factor, mode, padding_mode ):

    img = np.copy(image)

    h,w = img.shape[:2]
    img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=mode )
    img = cunsqueeze(img)
    hn, wn = img.shape[:2]

    borderX = float( abs(wn-w) )/2.0
    borderY = float( abs(hn-h) )/2.0

    padxL = int(np.floor( borderY ))
    padxR = int(np.ceil(  borderY ))
    padyT = int(np.floor( borderX ))
    padyB = int(np.ceil(  borderX ))

    if factor < 1:  img = cv2.copyMakeBorder(img, padxL, padxR, padyT, padyB, borderType=padding_mode)
    else: img = img[ padyT:padyT+h, padxL:padxL+w, : ]

    img = cunsqueeze(img)
    return img


def pad(image,  h_pad, w_pad, padding_mode):
    img = np.copy(image)
    height, width = img.shape[:2]
    img = cv2.copyMakeBorder(img, h_pad, h_pad, w_pad, w_pad, padding_mode)
    img = cunsqueeze(img)
    return img

def hflip( x ):
    return cunsqueeze( cv2.flip( x, 0 ) )

def vflip( x ):
    return cunsqueeze( cv2.flip( x, 1 )  )

def rotate90( x ):
    return cunsqueeze( cv2.flip( cunsqueeze(x).transpose(1,0,2),1) )

def rotate180( x ):
    return cunsqueeze( cv2.flip(x,-1) )

def rotate270( x ):
    return cunsqueeze( cv2.flip( cunsqueeze(x).transpose(1,0,2),0) )

def transpose( x ):
    return cunsqueeze( cunsqueeze(x).transpose(1,0,2) )



def is_box_inside(img, box ):
    return box[0] < 0 or box[1] < 0 or box[2]+box[0] >= img.shape[1] or box[3]+box[1] >= img.shape[0]

def pad_img_to_fit_bbox(image, box, padding_mode):

    img = np.copy(image)
    x1,y1,x2,y2 = box
    x2 = x1+x2; y2 = y1+y2

    padxL = np.abs(np.minimum(0, y1))
    padxR = np.maximum(y2 - img.shape[0], 0)
    padyT = (np.abs(np.minimum(0, x1)))
    padyB = np.maximum(x2 - img.shape[1], 0)
    img = cv2.copyMakeBorder(img, padxL, padxR, padyT, padyB, borderType=padding_mode )

    # img = np.pad(
    #     img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
    #           (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)),
    #           (0,0)), mode="constant")

    img = cunsqueeze(img)
    y2 += np.abs(np.minimum(0, y1))
    y1 += np.abs(np.minimum(0, y1))
    x2 += np.abs(np.minimum(0, x1))
    x1 += np.abs(np.minimum(0, x1))

    return img, [ x1, y1, x2-x1, y2-y1 ]

def imcrop( image, box, padding_mode ):
    """ Image crop
    Args
        @image
        @box: [x,y,w,h]
    """
    img = np.copy(image)
    h, w, c = img.shape
    if is_box_inside(img, box):
        img, box = pad_img_to_fit_bbox(img, box, padding_mode)
    x, y, new_w, new_h = box
    imagecrop = img[y:y + new_h, x:x + new_w, : ]
    imagecrop = cunsqueeze(imagecrop)

    return imagecrop

def unsharp(image, size=9, strength=0.25, alpha=5 ):

    image = image.astype(np.float32)
    size  = 1+2*(int(size)//2)
    strength = strength*255
    blur  = cv2.GaussianBlur(image, (size,size), strength)
    image = alpha*image + (1-alpha)*blur
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image

def gaussian_noise(image, sigma=0.5):

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255

    H,W  = gray.shape
    noise = np.array([random.gauss(0,sigma) for i in range(H*W)])
    noise = noise.reshape(H,W)
    noisy = gray + noise
    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)

    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image

def speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = np.array([random.random() for i in range(H*W)])
    noise = noise.reshape(H,W)
    noisy = gray + gray * noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image

def inv_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = sigma*random.randn(H,W)
    noise = np.array([random.random() for i in range(H*W)])
    noise = noise.reshape(H,W)
    noisy = gray + (1-gray) * noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image

def get_elastic_transform(shape, size_grid, deform):
    """Get elastic tranform
    Args:
        @shape: image shape
        @size_grid: size of the grid (example (10,10) )
        @deform: deform coeficient
    """

    m,n=shape[:2]
    grid_x, grid_y = np.mgrid[:m,:n]

    source = []
    destination = []

    for i in range(int(m/size_grid)+1):
        for j in range(int(n/size_grid)+1):
            source = source + [np.array([i*size_grid, j*size_grid])]
            noisex = round(random.uniform(-deform,deform))
            noisey = round(random.uniform(-deform,deform))
            noise  = np.array( [noisex,noisey] )
            if i==0 or j==0 or i==int(m/size_grid) or j==int(n/size_grid): noise = np.array([0,0])
            destination = destination + [np.array([i*size_grid, j*size_grid])+noise ]

    source=np.vstack(source)
    destination=np.vstack(destination)
    destination[destination<0] = 0
    destination[destination>=n] = n-1

    grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')

    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(m,n)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(m,n)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    return map_x_32, map_y_32

def get_tensor_elastic_transform( shape, size_grid, deform):
    """Get elastic tranform for tensor
    Args:
        @shape: image shape
        @size_grid: size of the grid (example (10,10) )
        @deform: deform coeficient
    """
    target_height, target_width = shape[:2]

    target_control_points = torch.Tensor( list(itertools.product(
        torch.arange(-1.0, 1.00001, 2.0 / (size_grid-1)),
        torch.arange(-1.0, 1.00001, 2.0 / (size_grid-1)),
        )))

    source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-deform, deform)
    tps = TPSGridGen(target_height, target_width, target_control_points)
    source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))
    grid = source_coordinate.view(1, target_height, target_width, 2)

    return grid

def get_geometric_random_transform( imsize, degree, translation, warp ):
    """Transform the image for data augmentation
    Args:
        @degree: Max rotation angle, in degrees. Direction of rotation is random.
        @translation: Max translation amount in both x and y directions,
            expressed as fraction of total image width/height
        @warp: Max warp amount for each of the 3 reference points,
            expressed as fraction of total image width/height

    """

    height, width = imsize[:2]
    degree = degree * math.pi / 180

    # Rotation
    center = (width//2, height//2)
    theta = random.uniform(-degree, degree)
    rotation_mat = cv2.getRotationMatrix2D(center, -theta*180/math.pi, 1)

    # Translation
    x_offset = translation * width * random.uniform(-1, 1)
    y_offset = translation * height * random.uniform(-1, 1)
    translation_mat = np.float32( np.array([[1, 0, x_offset], [0, 1, y_offset]]) )

    # # Warp
    # # NOTE: The commented code below is left for reference
    # # The warp function tends to blur the image, so it is not useds
    src_triangle = np.float32([[0, 0], [0, height], [width, 0]])
    x_offsets = [warp * width * random.uniform(-1, 1) for _ in range(3)]
    y_offsets = [warp * height * random.uniform(-1, 1) for _ in range(3)]
    dst_triangle = np.float32([[x_offsets[0], y_offsets[0]],\
                             [x_offsets[1], height + y_offsets[1]],\
                             [width + x_offsets[2], y_offsets[2]]])
    warp_mat = cv2.getAffineTransform(src_triangle, dst_triangle)


    return rotation_mat, translation_mat, warp_mat

def applay_geometrical_transform( image, mat_r, mat_t, mat_w, interpolate_mode, padding_mode  ):
    h,w = image.shape[:2]
    image = cv2.warpAffine(image, mat_r, (w,h), flags=interpolate_mode, borderMode=padding_mode )
    image = cv2.warpAffine(image, mat_t, (w,h), flags=interpolate_mode, borderMode=padding_mode )
    image = cv2.warpAffine(image, mat_w, (w,h), flags=interpolate_mode, borderMode=padding_mode )
    image = cunsqueeze(image)
    return image

def square_resize(img, newsize, interpolate_mode, padding_mode):

    image = np.copy(img)
    w, h, channels = image.shape;
    if w == h: return image

    if w>h:
        padxL = int(np.floor( (w-h) / 2.0));
        padxR = int(np.ceil( (w-h) / 2.0)) ;
        padyT, padyB = 0,0
    else:
        padxL, padxR = 0,0;
        padyT = int(np.floor( (h-w) / 2.0));
        padyB = int(np.ceil( (h-w) / 2.0));

    image = cv2.copyMakeBorder(image, padxL, padxR, padyT, padyB, borderType=padding_mode)
    image = cv2.resize(image, (newsize, newsize) , interpolation = interpolate_mode)

    image = cunsqueeze(image)
    return image

def draw_grid(imgrid, grid_size=50, color=(255,0,0), thickness=1):

    m,n = imgrid.shape[:2]
    # Draw grid lines
    for i in range(0, n-1, grid_size):
        cv2.line(imgrid, (i+grid_size, 0), (i+grid_size, m), color=color, thickness=thickness)
    for j in range(0, m-1, grid_size):
        cv2.line(imgrid, (0, j+grid_size), (n, j+grid_size), color=color, thickness=thickness)
    return imgrid

def resize_unet_transform(img, size, interpolate_mode, padding_mode):

    image = np.copy(img)
    height, width, ch = image.shape

    #unet required input size
    downsampleFactor = 16
    d4a_size   = 0
    padInput   = (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2
    padOutput  = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2
    d4a_size   = math.ceil( (size - padOutput)/downsampleFactor)
    input_size  = downsampleFactor*d4a_size + padInput
    output_size = downsampleFactor*d4a_size + padOutput;

    if height < width:
        asp = float(height)/width
        w = output_size
        h = int(w*asp)
    else:
        asp = float(width)/height
        h = output_size
        w = int(h*asp)

    #resize mantaining aspect ratio
    image = cv2.resize(image, (w,h), interpolation = interpolate_mode)
    image = cunsqueeze(image)

    borderX = float(input_size-w)/2.0
    borderY = float(input_size-h)/2.0

    padxL = int(np.floor( borderY ))
    padxR = int(np.ceil(  borderY ))
    padyT = int(np.floor( borderX ))
    padyB = int(np.ceil(  borderX ))

    #image = square_resize(image, input_size, interpolate_mode, padding_mode)
    image = cv2.copyMakeBorder(image, padxL, padxR, padyT, padyB, borderType=padding_mode)
    image = cv2.resize(image, (input_size, input_size) , interpolation = interpolate_mode)
    image = cunsqueeze(image)

    return image

def ffftshift2(h):
    H = np.fft.fft2(h)
    H = np.abs( np.fft.fftshift( H ) )
    return H

def norm_fro(a,b):
    return np.sum( (a-b)**2.0 );

def complex2vector(c):
    '''complex to vector'''
    return np.concatenate( ( c.real, c.imag ) , axis=1 )

def image_to_array(image, channels=None):
    """
    Returns an image as a np.array

    Arguments:
    image -- a PIL.Image or numpy.ndarray

    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    """

    if channels not in [None, 1, 3, 4]:
        raise ValueError('unsupported number of channels: %s' % channels)

    if isinstance(image, PIL.Image.Image):
        # Convert image mode (channels)
        if channels is None:
            image_mode = image.mode
            if image_mode not in ['L', 'RGB', 'RGBA']:
                raise ValueError('unknown image mode "%s"' % image_mode)
        elif channels == 1:
            # 8-bit pixels, black and white
            image_mode = 'L'
        elif channels == 3:
            # 3x8-bit pixels, true color
            image_mode = 'RGB'
        elif channels == 4:
            # 4x8-bit pixels, true color with alpha
            image_mode = 'RGBA'
        if image.mode != image_mode:
            image = image.convert(image_mode)
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.reshape(image.shape[:2])
        if channels is None:
            if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in [3, 4])):
                raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 1:
            if image.ndim != 2:
                if image.ndim == 3 and image.shape[2] in [3, 4]:
                    # color to grayscale. throw away alpha
                    image = np.dot(image[:, :, :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                else:
                    raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 3:
            if image.ndim == 2:
                # grayscale to color
                image = np.repeat(image, 3).reshape(image.shape + (3,))
            elif image.shape[2] == 4:
                # throw away alpha
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 4:
            if image.ndim == 2:
                # grayscale to color
                image = np.repeat(image, 4).reshape(image.shape + (4,))
                image[:, :, 3] = 255
            elif image.shape[2] == 3:
                # add alpha
                image = np.append(image, np.zeros(image.shape[:2] + (1,), dtype='uint8'), axis=2)
                image[:, :, 3] = 255
            elif image.shape[2] != 4:
                raise ValueError('invalid image shape: %s' % (image.shape,))
    else:
        raise ValueError('resize_image() expected a PIL.Image.Image or a numpy.ndarray')

    return image

def resize_image( img, height, width,
                 resize_mode=None,
                 padding_mode=cv2.BORDER_CONSTANT,
                 interpolate_mode=cv2.INTER_LINEAR,
                 ):
    """
    Resizes an image and returns it as a np.array

    Arguments:
    image --  numpy.ndarray
    height -- height of new image
    width -- width of new image

    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    resize_mode -- can be crop, squash, fill or half_crop
    """

    if resize_mode is None:
        resize_mode = 'squash'
    if resize_mode not in ['crop', 'squash', 'fill', 'half_crop', 'asp', 'square']:
        raise ValueError('resize_mode "%s" not supported' % resize_mode)

    # convert to array
    image = cunsqueeze( np.copy(img) )
    #image = image_to_array(image, channels)


    # No need to resize
    if image.shape[0] == height and image.shape[1] == width:
        return image

    # Resize
    #interp = 'bilinear'

    width_ratio  = float(image.shape[1]) / width
    height_ratio = float(image.shape[0]) / height

    if resize_mode == 'squash' or width_ratio == height_ratio:
        image = cv2.resize(image, (width, height) , interpolation = interpolate_mode)
        image = cunsqueeze(image)
        return image

    elif resize_mode == 'asp':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = height
            resize_width = int(round(image.shape[1] / height_ratio))
        else:
            resize_width = width
            resize_height = int(round(image.shape[0] / width_ratio))
        image = cv2.resize(image, (resize_width, resize_height) , interpolation = interpolate_mode)
        image = cunsqueeze(image)
        return image

    elif resize_mode == 'square':
        w, h, channels = image.shape;
        if w == h: return image
        if w>h:
            padxL = int(np.floor( (w-h) / 2.0));
            padxR = int(np.ceil( (w-h) / 2.0)) ;
            padyT, padyB = 0,0
        else:
            padxL, padxR = 0,0;
            padyT = int(np.floor( (h-w) / 2.0));
            padyB = int(np.ceil( (h-w) / 2.0));

        image = cv2.copyMakeBorder(image, padyT, padyB, padxL, padxR, borderType=padding_mode)
        image = cv2.resize(image, (width, height) , interpolation = interpolate_mode)
        image = cunsqueeze(image)
        return image

    elif resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = height
            resize_width = int(round(image.shape[1] / height_ratio))
        else:
            resize_width = width
            resize_height = int(round(image.shape[0] / width_ratio))
        image = cv2.resize(image, (resize_width, resize_height) , interpolation = interpolate_mode)
        image = cunsqueeze(image)


        # chop off ends of dimension that is still too long
        if width_ratio > height_ratio:
            start = int(round((resize_width - width) / 2.0))
            return cunsqueeze(image[:, start:start + width, :])
        else:
            start = int(round((resize_height - height) / 2.0))
            return cunsqueeze(image[start:start + height, :, :])
    else:
        if resize_mode == 'fill':
            # resize to biggest of ratios (relatively smaller image), keeping aspect ratio
            if width_ratio > height_ratio:
                resize_width = width
                resize_height = int(round(image.shape[0] / width_ratio))
                if (height - resize_height) % 2 == 1:
                    resize_height += 1
            else:
                resize_height = height
                resize_width = int(round(image.shape[1] / height_ratio))
                if (width - resize_width) % 2 == 1:
                    resize_width += 1
            image = cv2.resize(image, (resize_width, resize_height) , interpolation = interpolate_mode)
            image = cunsqueeze(image)

        elif resize_mode == 'half_crop':
            # resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = int(round(image.shape[1] / new_ratio))
            resize_height = int(round(image.shape[0] / new_ratio))
            if width_ratio > height_ratio and (height - resize_height) % 2 == 1:
                resize_height += 1
            elif width_ratio < height_ratio and (width - resize_width) % 2 == 1:
                resize_width += 1
            image = cv2.resize(image, (resize_width, resize_height) , interpolation = interpolate_mode)
            image = cunsqueeze(image)

            # chop off ends of dimension that is still too long
            if width_ratio > height_ratio:
                start = int(round((resize_width - width) / 2.0))
                image = image[:, start:start + width]
            else:
                start = int(round((resize_height - height) / 2.0))
                image = image[start:start + height, :]
        else:
            raise Exception('unrecognized resize_mode "%s"' % resize_mode)

        # fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio:
            padding = (height - resize_height) / 2
            noise_size = (padding, width)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=0)
        else:
            padding = (width - resize_width) / 2
            noise_size = (height, padding)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=1)

        return cunsqueeze(image)

def compute_norm_mat(base_width, base_height):
    # normalization matrix used in image pre-processing
    x      = np.arange(base_width)
    y      = np.arange(base_height)
    X, Y   = np.meshgrid(x, y)
    X      = X.flatten()
    Y      = Y.flatten()
    A      = np.array([X*0+1, X, Y]).T
    A_pinv = np.linalg.pinv(A)
    return A, A_pinv

def equalization(image, A, A_pinv):

    image_new = []
    image = image.transpose( (2,0,1) )

    for img in image:

        # compute image histogram
        img_flat = img.flatten()
        img_hist = np.bincount(img_flat, minlength = 256)

        # cumulative distribution function
        cdf = img_hist.cumsum()
        cdf = cdf * (2.0 / cdf[-1]) - 1.0 # normalize

        # histogram equalization
        img_eq = cdf[img_flat]

        diff = img_eq - np.dot(A, np.dot(A_pinv, img_eq))

        # after plane fitting, the mean of diff is already 0
        std = np.sqrt(np.dot(diff,diff)/diff.size)
        if std > 1e-6:
            diff = diff/std
        img = diff.reshape(img.shape)
        image_new.append(img)

    return np.stack(image_new, 0).transpose( (1,2,0) )

def distort_img(img, roi, out_width, out_height, max_shift, max_scale, max_angle, max_skew, flip=True):
    shift_y = out_height*max_shift*rnd.uniform(-1.0,1.0)
    shift_x = out_width*max_shift*rnd.uniform(-1.0,1.0)

    # rotation angle
    angle = max_angle*rnd.uniform(-1.0,1.0)

    #skew
    sk_y = max_skew*rnd.uniform(-1.0, 1.0)
    sk_x = max_skew*rnd.uniform(-1.0, 1.0)

    # scale
    scale_y = rnd.uniform(1.0, max_scale)
    if rnd.choice([True, False]):
        scale_y = 1.0/scale_y
    scale_x = rnd.uniform(1.0, max_scale)
    if rnd.choice([True, False]):
        scale_x = 1.0/scale_x
    T_im = crop_img(img, roi, out_width, out_height, shift_x, shift_y, scale_x, scale_y, angle, sk_x, sk_y)
    if flip and rnd.choice([True, False]):
        T_im = np.fliplr(T_im)
    return T_im

def crop_img(img, roi, crop_width, crop_height, shift_x, shift_y, scale_x, scale_y, angle, skew_x, skew_y):
    # current face center
    ctr_in = np.array((roi.center().y, roi.center().x))
    ctr_out = np.array((crop_height/2.0+shift_y, crop_width/2.0+shift_x))
    out_shape = (crop_height, crop_width)
    s_y = scale_y*(roi.height()-1)*1.0/(crop_height-1)
    s_x = scale_x*(roi.width()-1)*1.0/(crop_width-1)

    # rotation and scale
    ang = angle*np.pi/180.0
    transform = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    transform = transform.dot(np.array([[1.0, skew_y], [0.0, 1.0]]))
    transform = transform.dot(np.array([[1.0, 0.0], [skew_x, 1.0]]))
    transform = transform.dot(np.diag([s_y, s_x]))
    offset = ctr_in-ctr_out.dot(transform)

    # each point p in the output image is transformed to pT+s, where T is the matrix and s is the offset
    T_im = ndimage.interpolation.affine_transform(input = img,
                                                  matrix = np.transpose(transform),
                                                  offset = offset,
                                                  output_shape = out_shape,
                                                  order = 1,   # bilinear interpolation
                                                  mode = 'reflect',
                                                  prefilter = False)
    return T_im


# Object transfrmation

def resize_box(box, fx, fy, ofx=0, ofy=0):
    box[:,0] = box[:,0]*fx + ofx
    box[:,1] = box[:,1]*fy + ofy
    box[:,2] = box[:,2]*fx + ofx
    box[:,3] = box[:,3]*fy + ofy
    return box

def resize_image_box(
        img,
        boxs,
        height, width,
        resize_mode=None,
        padding_mode=cv2.BORDER_CONSTANT,
        interpolate_mode=cv2.INTER_LINEAR,
        ):
    """
    Resizes an image and returns it as a np.array

    Arg:
        image --  numpy.ndarray
        box   --  numpy.array [x1, y1, x2, y2]
        height -- height of new image
        width -- width of new image

    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    resize_mode -- can be crop, squash, fill or half_crop
    """

    if resize_mode is None:
        resize_mode = 'squash'
    if resize_mode not in ['crop', 'squash', 'fill', 'half_crop', 'asp', 'square']:
        raise ValueError('resize_mode "%s" not supported' % resize_mode)

    # convert to array
    image = np.copy(img)
    box   = np.copy(boxs)
    h,w = image.shape[:2]

    # No need to resize
    if h == height and w == width:
        return image, box

    fx = float(w) / width
    fy = float(h) / height

    if resize_mode == 'squash' or fx == fy:

        image = cv2.resize(image, (width, height) , interpolation = interpolate_mode)
        image = cunsqueeze(image)
        return image, resize_box(box, 1/fx, 1/fy)

    elif resize_mode == 'asp':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if fx > fy:
            width  = int(round(w / fy))
        else:
            height = int(round(h / fy))

        image = cv2.resize(image, (width, height) , interpolation = interpolate_mode)
        image = cunsqueeze(image)
        fx = float(w) / width
        fy = float(h) / height
        return image, resize_box(box, 1/fx, 1/fy)


    elif resize_mode == 'square':

        if w != h:
            if w>h:
                padxL = int(np.floor( (w-h) / 2.0));
                padxR = int(np.ceil( (w-h) / 2.0)) ;
                padyT, padyB = 0,0
                fy=fx
            else:
                padxL, padxR = 0,0;
                padyT = int(np.floor( (h-w) / 2.0));
                padyB = int(np.ceil( (h-w) / 2.0));
                fx=fy

            image = cv2.copyMakeBorder(image, padxL, padxR, padyT, padyB, borderType=padding_mode)
            image = cv2.resize(image, (width, height) , interpolation = interpolate_mode)
            image = cunsqueeze(image)

        box = resize_box(box, 1, 1, padyT, padxL)
        box = resize_box(box, 1/fx, 1/fy, 0 , 0)
        return image, box

    elif resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if fx > fy:
            resize_height = height
            resize_width = int(round(image.shape[1] / fy))
        else:
            resize_width = width
            resize_height = int(round(image.shape[0] / fx))

        image = cv2.resize(image, (resize_width, resize_height) , interpolation = interpolate_mode)
        image = cunsqueeze(image)


        # chop off ends of dimension that is still too long
        if fx > fy:
            fx = float(w) / resize_width
            fy = float(h) / resize_height
            start = int(round((resize_width - width) / 2.0))
            return image[:, start:start + width], resize_box(box, 1/fx, 1/fy, -start, 0  )
        else:
            fx = float(w) / resize_width
            fy = float(h) / resize_height
            start = int(round((resize_height - height) / 2.0))
            return image[start:start + height, :], resize_box(box, 1/fx, 1/fy, 0, -start  )

    else:
        raise Exception('unrecognized resize_mode "%s"' % resize_mode)


    return image, box
