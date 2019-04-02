
#STD MODULES
import os
import numpy as np
import math
import shutil

#TORCH MODULES
import torch
import torch.nn as nn
import torch.nn.functional as F

#LOCAL MODULES
from . import netlearningrate
from . import graphic as gph
from . import utils

#----------------------------------------------------------------------------------------------
# Neural Net Abstract Class

class NeuralNetAbstract(object):
    """
    Abstract Neural Net 
    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0
        ):
        """
        Initialization
        Args:
            @patchproject (str): path project
            @nameproject (str):  name project
            @no_cuda (bool): system cuda (default is True)
            @parallel (bool)
            @seed (int)
            @print_freq (int)
            @gpu (int)
        """

        # cuda
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.parallel = not no_cuda and parallel
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.set_device( gpu )
            torch.cuda.manual_seed(seed)

        # set project directory
        self.nameproject = nameproject
        self.pathproject = os.path.join(patchproject, nameproject)

        # Set the graphic visualization
        self.plotter = None
        #self.plotter = gph.VisdomLinePlotter(env_name=nameproject)
        
        # initialization var
        self.print_freq = print_freq
        self.num_input_channels = 0
        self.num_output_channels = 0
        self.size_input = 0
        self.lr = 0.0001
        self.start_epoch = 0        

        self.s_arch = ''
        self.s_optimizer = ''
        self.s_lerning_rate_sch = ''
        self.s_loss = ''

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.lrscheduler = None
        self.vallosses = 0

    def create(self, 
        arch, 
        num_output_channels, 
        num_input_channels, 
        loss, 
        lr, 
        optimizer, 
        lrsch, 
        pretrained=False,
        cfg_model=None,
        cfg_loss=None,
        cfg_opt=None,
        cfg_scheduler=None,
        ):
        """
        Create 
        Args:
            @arch (string): architecture
            @loss (string):
            @lr (float): learning rate
            @optimizer (string) : 
            @lrsch (string): scheduler learning rate
            @pretrained (bool)
        """
                
        self.s_arch = arch
        self.s_optimizer = optimizer
        self.s_lerning_rate_sch = lrsch
        self.s_loss = loss

        if cfg_model is None:       #defaul configurate
            cfg_model = {}
        if cfg_loss is None:        #defaul configurate
            cfg_loss = {}
        if cfg_opt is None:         #defaul configurate
            cfg_opt = {}
        if cfg_scheduler is None:   #defaul configurate
            cfg_scheduler = {}

        # create project directory
        self.pathmodels = os.path.join(self.pathproject, 'models')
        if not os.path.exists(self.pathproject):
            os.makedirs(self.pathproject)
        if not os.path.exists(self.pathmodels):
            os.makedirs(self.pathmodels)
        
        # create visual visdom plot
        self.plotter =  gph.VisdomLinePlotter(env_name=self.nameproject)
                
        self._create_model( arch, num_output_channels, num_input_channels, pretrained, **cfg_model )
        self._create_loss( loss, **cfg_loss )
        self._create_optimizer( optimizer, lr, **cfg_opt )
        self._create_scheduler_lr( lrsch, **cfg_scheduler )

    def training(self, data_loader, epoch=0):
        pass

    def evaluate(self, data_loader, epoch=0):
        pass

    def test(self, data_loader):
        pass

    def inference(self, image):        
        pass

    def representation(self, data_loader):
        pass
    
    def fit( self, train_loader, val_loader, epochs=100, snapshot=10 ):

        best_prec = 0
        print('\nEpoch: {}/{}(0%)'.format(self.start_epoch, epochs))
        print('-' * 25)

        self.evaluate(val_loader, epoch=self.start_epoch)        
        for epoch in range(self.start_epoch, epochs):       

            try:
                
                self._to_beging_epoch(epoch, epochs, train_loader, val_loader)

                self.adjust_learning_rate(epoch)     
                self.training(train_loader, epoch)

                print('\nEpoch: {}/{} ({}%)'.format(epoch,epochs, int((float(epoch)/epochs)*100) ) )
                print('-' * 25)
                
                prec = self.evaluate(val_loader, epoch+1 )            

                # remember best prec@1 and save checkpoint
                is_best = prec > best_prec
                best_prec = max(prec, best_prec)
                if epoch % snapshot == 0 or is_best or epoch==(epochs-1) :
                    self.save(epoch, best_prec, is_best, 'chk{:06d}.pth.tar'.format(epoch))

                self._to_end_epoch(epoch, epochs, train_loader, val_loader)

            except KeyboardInterrupt:
                
                print('Ctrl+C, saving snapshot')
                is_best = False
                best_prec = 0
                self.save(epoch, best_prec, is_best, 'chk{:06d}.pth.tar'.format(epoch))
                return

    def _to_beging_epoch(self, epoch, epochs, train_loader, val_loader, **kwargs):
        pass

    def _to_end_epoch(self, epoch, epochs, train_loader, val_loader, **kwargs):
        pass


    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained):
        """
        Create model
        Args:
            @arch (string): select architecture
            @num_output_channels (int)
            @num_input_channels (int)
            @pretrained (bool)
        """    
        pass

    def _create_loss(self, loss, **kwargs):
        """
        Create loss
        Args:
            @loss (string): select loss function
        """
        pass

    def _create_optimizer(self, optimizer='adam', lr=0.0001, **kwargs ):
        """
        Create optimizer
        Args:
            @optimizer (string): select optimizer function
            @lr (float): learning rate
            @momentum (float): momentum
        """
        
        self.optimizer = None

        # create optimizer
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam( self.net.parameters(), lr=lr, amsgrad=True )  
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD( self.net.parameters(), lr=lr, **kwargs ) #momentum=0.9, weight_decay=5e-4 
        elif optimizer == 'rprop':
            self.optimizer = torch.optim.Rprop( self.net.parameters(), lr=lr) 
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop( self.net.parameters(), lr=lr)           
        else:
            assert(False)

        self.lr = lr; 
        self.s_optimizer = optimizer

    def _create_scheduler_lr(self, lrsch, **kwargs ):
        
        #MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        #ExponentialLR
        #CosineAnnealingLR

        self.lrscheduler = None

        if lrsch == 'fixed':
            pass           
        elif lrsch == 'step':
            self.lrscheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **kwargs )# step_size=3, gamma=0.1 
        elif lrsch == 'cyclic': 
            self.lrscheduler = netlearningrate.CyclicLR(self.optimizer, **kwargs)
        elif lrsch == 'exp':
            self.lrscheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, **kwargs ) # gamma=0.99 
        elif lrsch == 'plateau':
            self.lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **kwargs ) # 'min', patience=10
        else:
            assert(False)
        
        self.s_lerning_rate_sch = lrsch

    def adjust_learning_rate(self, epoch ):
        """
        Update learning rate
        """       
 
        # update
        if self.s_lerning_rate_sch == 'fixed': 
            lr = self.lr
        elif self.s_lerning_rate_sch == 'plateau':
            self.lrscheduler.step( self.vallosses )
            for param_group in self.optimizer.param_groups:
                lr = float(param_group['lr'])
                break            
        else:                    
            self.lrscheduler.step() 
            lr = self.lrscheduler.get_lr()[0]        

        # draw
        self.plotter.plot('lr', 'learning rate', epoch, lr )
 
    def resume(self, pathnammodel):
        """
        Resume: optionally resume from a checkpoint
        """ 
        net = self.net.module if self.parallel else self.net
        start_epoch, prec = utils.resumecheckpoint( 
            pathnammodel, 
            net, 
            self.optimizer 
            )

        self.start_epoch = start_epoch
        return start_epoch, prec

    def save(self, epoch, prec, is_best=False, filename='checkpoint.pth.tar'):
        """
        Save model
        """
        print('>> save model epoch {} ({}) in {}'.format(epoch, prec, filename))
        net = self.net.module if self.parallel else self.net
        utils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': self.s_arch,
                'imsize': self.size_input,
                'num_classes': self.num_output_channels,
                'num_channels': self.num_input_channels,
                'state_dict': net.state_dict(),
                'prec': prec,
                'optimizer' : self.optimizer.state_dict(),
            }, 
            is_best,
            self.pathmodels,
            filename
            )
   
    def load(self, pathnamemodel):
        bload = False
        if pathnamemodel:
            if os.path.isfile(pathnamemodel):
                print("=> loading checkpoint '{}'".format(pathnamemodel))
                checkpoint = torch.load( pathnamemodel ) if self.cuda else torch.load( pathnamemodel, map_location=lambda storage, loc: storage )
                
                self._create_model(checkpoint['arch'], checkpoint['num_classes'], checkpoint['num_channels'], False )      
                self.size_input = checkpoint['imsize']         
                self.net.load_state_dict( checkpoint['state_dict'] )               

                print("=> loaded checkpoint for {} arch!".format(checkpoint['arch']))
                bload = True
            else:
                print("=> no checkpoint found at '{}'".format(pathnamemodel))        
        return bload
   
    def __str__(self): 
        return str(
                'Name: {} \n'
                'arq: {} \n'
                'loss: {} \n'
                'optimizer: {} \n'
                'lr: {} \n'
                'size input: {} \n'
                'num input channels {} \n'
                'num output channels: {} \n'
                'Model: \n{} \n'.format(
                    self.nameproject,
                    self.s_arch,
                    self.s_loss,
                    self.s_optimizer,
                    self.lr,
                    self.size_input,
                    self.num_input_channels,
                    self.num_output_channels,
                    self.net
                    )
                )


