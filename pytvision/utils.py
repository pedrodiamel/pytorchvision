

import os
import shutil
import numpy as np
import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import MiniBatchKMeans

import warnings
warnings.filterwarnings("ignore")


def to_one_hot(mask, size):    
    n, c, h, w = size
    ymask = torch.FloatTensor(size).zero_()
    new_mask = torch.LongTensor(n,1,h,w)
    if mask.is_cuda:
        ymask = ymask.cuda(mask.get_device())
        new_mask = new_mask.cuda(target.get_device())
    new_mask[:,0,:,:] = torch.clamp(mask.data, 0, c-1)
    ymask.scatter_(1, new_mask , 1.0)    
    return Variable(ymask)

def centercrop(image, w, h):        
    nt, ct, ht, wt = image.size()
    padw, padh = (wt-w) // 2 ,(ht-h) // 2
    if padw>0 and padh>0: image = image[:,:, padh:-padh, padw:-padw]
    return image

def flatten(x):        
    x_flat = x.clone()
    x_flat = x_flat.view(x.shape[0], -1)
    return x_flat

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x, cuda, requires_grad=False, volatile=False):
    if cuda: x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def argmax(x): return torch.max(x, 1)[1]

def fit(net, ngpu, inputs):
    if ngpu > 1: outputs = nn.parallel.data_parallel(net, inputs, range(ngpu))
    else: outputs = net(inputs)
    return outputs

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))

def resumecheckpoint(resume, net, optimizer):
    """Optionally resume from a checkpoint"""
    start_epoch = 0
    prec = 0
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            prec = checkpoint['prec']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    return start_epoch, prec


def sigmoid(x):
    return 1. / (1 + np.exp(-x))
 
def normalize(data):
    data = data - np.min(data)
    data = data / np.max(data)  
    return data

def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)
    if not masks: return [labeled]
    else: return masks


def quantized(imagein, k=5):
    
    h,w = imagein.shape[:2]
    image = cv2.cvtColor(imagein, cv2.COLOR_RGB2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters =  k )
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)

    return quant



#---------------------------------------------------------------------------
# RunLen code and decoder 

# def rle_encode(img):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     pixels = img.flatten()
#     pixels[0] = 0
#     pixels[-1] = 0
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
#     runs[1::2] = runs[1::2] - runs[:-1:2]
#     return ' '.join(str(x) for x in runs)


def rle_encode(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    if len(rle) != 0 and rle[-1] + rle[-2] == x.size:
        rle[-2] = rle[-2] - 1

    return rle

# def run_decode(rle, H, W, fill_value=255):
    
#     mask = np.zeros((H * W), np.uint8)
#     rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
#     for r in rle:
#         start = r[0]-1
#         end = start + r[1]
#         mask[start : end] = fill_value
#     mask = mask.reshape(W, H).T # H, W need to swap as transposing.
#     return mask


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle #mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
