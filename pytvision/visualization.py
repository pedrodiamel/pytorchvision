

import numpy as np

#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import transforms, utils
from graphviz import Digraph

import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def draw_bounding_box(image, label,  color='red', thickness=4):
    
    bbox = label.bbox;
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB') 
    im_width, im_height = image_pil.size

    draw = ImageDraw.Draw(image_pil)
    xmin = bbox[0,0]; ymin = bbox[0,1];
    xmax = bbox[1,0]; ymax = bbox[1,1];
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
    
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/ttf-bitstream-vera/Vera.ttf', 32) #'arial.ttf'
    except IOError:
        font = ImageFont.load_default()
    
    text_bottom = top
    
    # Reverse list and print from bottom to top.
    #for display_str in display_str_list[::-1]:
    
    display_str = '{}({})  '.format(label.classe, label.score);
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), 
        (left + text_width, text_bottom)],
        fill=color)
    
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin
    
    np.copyto(image, np.array(image_pil))

def draw_bounding_box_dic(image, label,  color='red', thickness=4):
    
    bbox = label['bbox'];
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB') 
    im_width, im_height = image_pil.size

    draw = ImageDraw.Draw(image_pil)
    xmin = bbox[0,0]; ymin = bbox[0,1];
    xmax = bbox[1,0]; ymax = bbox[1,1];
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
    
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/ttf-bitstream-vera/Vera.ttf', 32) #'arial.ttf'
    except IOError:
        font = ImageFont.load_default()
    
    text_bottom = top
    
    # Reverse list and print from bottom to top.
    #for display_str in display_str_list[::-1]:
    
    display_str = '{}({})  '.format(label['classe'], label['score']);
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), 
        (left + text_width, text_bottom)],
        fill=color)
    
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin
    
    np.copyto(image, np.array(image_pil))


def plotboxcv(image,bbox,color=(0,255,0)):
    image1 = image.copy()
    x = bbox[0,:]; y = bbox[1,:];
    cv2.rectangle(image1,(int(bbox[0,0]),int(bbox[0,1])),(int(bbox[1,0]),int(bbox[1,1]) ),color,4)
    return image1




def torgb(im):
    if len(im.shape)==2:
        im = np.expand_dims(im, axis=2) 
        im = np.concatenate( (im,im,im), axis=2 )
    return im
        
def setcolor(im, mask, color):    
    tmp=im.copy()
    tmp=np.reshape( tmp, (-1, im.shape[2])  )   
    mask = np.reshape( mask, (-1,1))      
    tmp[ np.where(mask>0)[0] ,:] = color
    im=np.reshape( tmp, (im.shape)  )
    return im

def lincomb(im1,im2,mask, alpha):
    
    #im = np.zeros( (im1.shape[0], im1.shape[1], 3) )
    im = im1.copy()       
    row, col = np.where(mask>0)
    for i in range( len(row) ):
        r,c = row[i],col[i]
        #print(r,c)
        im[r,c,0] = im1[r,c,0]*(1-alpha) + im2[r,c,0]*(alpha)
        im[r,c,1] = im1[r,c,1]*(1-alpha) + im2[r,c,1]*(alpha)
        im[r,c,2] = im1[r,c,2]*(1-alpha) + im2[r,c,2]*(alpha)
    return im

def makebackgroundcell(labels):
    ch = labels.shape[2]
    cmap = plt.get_cmap('jet_r')
    imlabel = np.zeros( (labels.shape[0], labels.shape[1], 3) )    
    for i in range(ch):
        mask  = labels[:,:,i]
        color = cmap(float(i)/ch)
        imlabel = setcolor(imlabel,mask,color[:3])
    return imlabel

def makeedgecell(labels):
    ch = labels.shape[2]
    cmap = plt.get_cmap('jet_r')
    imedge = np.zeros( (labels.shape[0], labels.shape[1], 3) )    
    for i in range(ch):
        mask  = labels[:,:,i]
        color = cmap(float(i)/ch)
        mask = mask.astype(np.uint8)
        contours,_ = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE )
        for cnt in contours: cv2.drawContours(imedge, cnt, -1, color[:3], 1)
    return imedge

def makeimagecell(image, labels, alphaback=0.3, alphaedge=0.3):
    
    imagecell = image.copy()
    imagecell = imagecell - np.min(imagecell)
    imagecell = imagecell / np.max(imagecell)    
    imagecell = torgb(imagecell)
     
    mask  = np.sum(labels, axis=2)
    imagecellbackground = makebackgroundcell(labels)
    imagecelledge = makeedgecell(labels)
    maskedge = np.sum(imagecelledge, axis=2)
    
    imagecell = lincomb(imagecell,imagecellbackground, mask, alphaback )
    imagecell = lincomb(imagecell,imagecelledge, maskedge, alphaedge )
            
    return imagecell

def display_samples(dataloader, row=3, col=3, alphaback=0.3, alphaedge=0.2):
    """
    Display random data from dataset
    For debug only
    """
    fig, ax = plt.subplots(row, col, figsize=(8, 8), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})
    
    for i in range(row):
        for j in range(col):
            
            k = np.random.randint( len(dataloader) )
            image, labels = dataloader[ k ]
            imagecell = makeimagecell(image, labels, alphaback=alphaback, alphaedge=alphaedge)
            
            ax[i,j].imshow(imagecell)
            ax[i,j].set_title('Image Idx: %d' % (k,))

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


def drawmanifold( images, x, S=2000, s=50, Ntake = 100, freq_print=100 ):
    '''
    drawmanifold
    S = 2000 # size of full embedding image
    s = 50 # size of every single image   
    Ntake = 200 #count of the object
    
    #Example
    
    Mf = drawmanifold(data, Xt)
    plt.figure( figsize=(12,12) )
    plt.imshow(Mf)
    plt.show()
    
    '''
    
    #normalization 
    x = x - np.min(x)
    x = x / np.max(x)
    
    G = np.zeros( (S, S, 3), dtype='uint8')
    for i in range(Ntake):

        if (i-1)%freq_print == 0:
            print('{}/{}...\n'.format(i, Ntake)  )

        # location
        a = np.ceil(x[i, 0] * (S-s))
        b = np.ceil(x[i, 1] * (S-s))
        
        a = int(a - (a)%s) 
        b = int(b - (b)%s) 

        if G[a,b,1] != 0:
            continue # spot already filled
        
        I,_ = images[i]; I = np.array(I)        
        if len(I.shape)==2: I = I[:,:, np.newaxis ]
        if I.shape[2] == 1: I = np.concatenate((I,I,I), axis=2 )           
        I = scipy.misc.imresize(I, (s,s), interp='bilinear' )   
        G[a:a+s, b:b+s, :] = I;
                      
    return G


# Helper function to show a batch
def show_limage_batch(sample_batched, nrow=8, labels=None):
    """Show image for a batch of samples."""
    images_batch, label_batch = sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    rows = np.min( (images_batch.shape[0]//nrow + 1, 64 )  )

    grid = utils.make_grid(images_batch, nrow=nrow, normalize=True)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
    strlabels = ''
    for i in range(batch_size):
        label = label_batch[i].numpy()
        if labels:
            strlabels +='{}, '.format( labels[ np.argmax( label ) ] )
        else: 
            strlabels +='{}, '.format( i )
        
    plt.title('Batch from dataloader') 
    print(strlabels)


def visualizatedataset( data, num=25, imsize=(64,64,3), padding=1 ):    
    """
    Visualization data set
        @data: data loader 
        @num: number of images show
    """
    
    databatch = torch.zeros( [num, imsize[2], imsize[0], imsize[1]], dtype=torch.float32 )
    for i in range(num): 
        idx = np.random.randint( len(data) )
        databatch[i,...] = data[idx]['image']        
    grid = utils.make_grid( databatch , nrow= int(np.sqrt(num)), normalize=True, padding=padding )       
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def savedataset( pathname, data, num=25, imsize=(64,64,3), padding=1 ):  
    databatch = torch.zeros( [num, imsize[2], imsize[0], imsize[1]], dtype=torch.float32 )
    for i in range(num): 
        idx = np.random.randint( len(data) )
        x = data[idx]['image']
        x = x - x.min()
        x = x / x.max()
        databatch[i,...] = x      
    utils.save_image(databatch, pathname, nrow=int(np.sqrt(num)) )   



def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    """
    vistensor: visuzlization tensor
        @ch: visualization channel 
        @allkernels: visualization all tensores
    """        
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

def savetensor(tensor, filename, ch=0, allkernels=False, nrow=8, padding=2):
    """
    savetensor: save tensor
        @filename: file name path
        @ch: visualization channel 
        @allkernels: visualization all tensores
    """    
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)    
    utils.save_image(tensor, filename, nrow=nrow )


def smooth(x,window_len=11,window='hanning'):    
    """
    http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        #assert all(isinstance(p, Variable) for p in params.values())        
        param_map = {id(v): k for k, v in params.items()}


    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                #name = param_map[id(u)] if params is not None else ''
                #node_name = '%s\n %s' % (name, size_to_str(u.size()))
                node_name = '%s\n %s' % (param_map.get(id(u.data)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
                
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot
