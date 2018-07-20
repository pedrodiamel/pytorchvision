

import numpy as np 
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from pytvision.transforms.ellipse_render import CircleRender, EllipseRender

def test_ellipserender():

    n = 512; m = 512; cnt = 5;
    rmin = 5; rmax = 50;
    border = 90;
    sigma = 0.2;
    #img, label, meta = CircleRender.generate( n, m, cnt, rmin, rmax, border, sigma, True) 
    img, label, meta = EllipseRender.generate( n, m, cnt, rmin, rmax, border, sigma, True) 

    plt.figure( figsize=(10,10) )
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(label.max(0))
    plt.show()


    print('DONE!!!') 

