import os
import sys
import numpy as np
import cv2


from skimage import io, transform, morphology, filters
from scipy import ndimage
import skimage.morphology as morph
import skfmm

#tranformations 

def isrgb( image ):
    return len(image.shape)==3 and image.shape[2]==3 

def to_rgb( image ):
    #to rgb
    if not isrgb( image ):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def to_gray( image ):
    if isrgb( image ):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def to_channels( image, ch ):
    if ch == 1:
        image = to_gray( image )[:,:,np.newaxis]
    elif ch == 3:
        image = to_rgb( image )
    else:
        assert(False)
    return image

def to_one_hot( x, nc ):    
    y = np.zeros((nc,1)); y[ int(x) ] = 1.0
    return y

def tolabel( x ):
    return (np.max(x,axis=0)>0) 

def summary(data):
    print(data.shape, data.min(), data.max())

def get_label_mask(mask_img, border_img, seed_ths, threshold, seed_size=8, obj_size=10):
    img_copy = np.copy(mask_img)
    m = img_copy * (1 - border_img)
    img_copy[m <= seed_ths] = 0
    img_copy[m > seed_ths] = 1
    img_copy = img_copy.astype(np.bool)
    img_copy = remove_small_objects(img_copy, seed_size).astype(np.uint8)
    mask_img[mask_img <= threshold] = 0
    mask_img[mask_img > threshold] = 1
    mask_img = mask_img.astype(np.bool)
    mask_img = remove_small_objects(mask_img, obj_size).astype(np.uint8)
    markers = ndimage.label(img_copy, output=np.uint32)[0]
    labels = watershed(mask_img, markers, mask=mask_img, watershed_line=True)
    return labels


def get_edges( masks ):
    edges = np.array([ morph.binary_dilation(get_contour(x)) for x in masks ])
    return edges  

def get_touchs( edges ):       
    A = np.array([ morph.binary_dilation( c, morph.square(3) )  for c in edges ]) 
    A = np.sum(A,axis=0)>1  
    I = morph.remove_small_objects( A, 3 )
    I = morph.skeletonize(I)
    I = morph.binary_dilation( I, morph.square(3) )    
    return I

def get_contour(img):    
    img = img.astype(np.uint8)
    edge = np.zeros_like(img)
    _,cnt,_ = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )
    cv2.drawContours( edge, cnt, -1, 1 , 1)
    edge = (edge>0).astype(np.uint8)    
    return edge

def get_center(img):
    cent = np.zeros_like(img).astype(np.uint8)
    y, x = ndimage.measurements.center_of_mass(img)
    cv2.circle(cent, (int(x), int(y)), 1, 1, -1)
    cent = (cent>0).astype(np.uint8) 

    cent  = np.array([ morph.binary_dilation(c) for c in cent ]) 
    cent = tolabel(cent) 
    return cent

def get_distance(x):
    return skfmm.distance((x).astype('float32') - 0.5) 

