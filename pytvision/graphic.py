from visdom import Visdom
import numpy as np
import cv2

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), 
                env=self.env, 
                opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epoch',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), 
                env=self.env, 
                win=self.plots[var_name], 
                name=split_name
                )



class VisdomScatter(object):
    """Scatter to Visdom"""
    
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.scatters = {}

    def scatter(self, X, Y, title, legend, markersize=10):        
        if title not in self.scatters: 
            self.scatters[title] = self.viz.scatter( X=X, Y=Y,
                    env=self.env,
                    opts=dict(
                    #legend=legend,
                    markersize=markersize,
                    title=title
                    )
                )
        else:            
            self.viz.scatter( X=X, Y=Y,
                    env=self.env,
                    win=self.scatters[title],
                    opts=dict(
                    #legend=legend,
                    markersize=markersize,
                    title=title
                    )
                )

class HeatMapVisdom(object):
    """Heat Map to Visdom"""
    
    def __init__(self, env_name='main', heatsize=None ):
        self.vis = Visdom()
        self.env = env_name
        self.hmaps = {}
        self.heatsize = heatsize
    
    def show(self, title, image):
        
        if self.heatsize:
            image = cv2.resize(image, self.heatsize, interpolation = cv2.INTER_LINEAR)

        if title not in self.hmaps:
            self.hmaps[title] = self.vis.heatmap(
                image, 
                env=self.env, 
                opts=dict(title=title)
            )
        else:
            self.vis.heatmap(
                image,
                env=self.env, 
                win=self.hmaps[title], 
                opts=dict(title=title)
            )
            
class ImageVisdom(object):
    """Images to Visdom"""
    
    def __init__(self, env_name='main', imsize=None):
        self.vis = Visdom()
        self.env = env_name
        self.images = {}
        self.imsize = imsize
    
    def show(self, title, image):
        
        if self.imsize:
            image = cv2.resize(image, self.imsize, interpolation = cv2.INTER_LINEAR)

        if title not in self.images:
            self.images[title] = self.vis.image(
                image, 
                env=self.env, 
                opts=dict(title=title)
            )
        else:
            self.vis.image(
                image,
                env=self.env, 
                win=self.images[title], 
                opts=dict(title=title)
            )