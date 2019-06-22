#----------------------------------------------------------------------------------------------
# DeXpression
# https://arxiv.org/abs/1509.05371
# Pedro D. Marrero Fernandez
#----------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DeXpression', 'dexpression']

model_urls = {
    'dexpression': '...',
}


def dexpression(pretrained=False, **kwargs):
    r"""DeXpression model architecture from the
    `"DeXpression: Deep ..." <https://arxiv.org/abs/1509.05371>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DeXpression(**kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['dexpressionnet']))
    return model


class DeXpression(nn.Module):
    
    def __init__(self, num_classes=8, num_channels=1 ):
        super(DeXpression, self).__init__()        
        
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.size_input=224 
        self.dim = 272*14*14

        self.conv01 = BasicConv2d(num_channels, 64, kernel_size=7, stride=2, padding=3)        
        self.pool01 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.featex01 = FeatExp(64)
        self.featex02 = FeatExp(272)        
        self.fc = nn.Linear(272*14*14, num_classes)
        nn.init.xavier_normal(self.fc.weight)

    def forward(self, x):       
        x = self.conv01(x)
        x = self.pool01(x)
        x = self.featex01(x)
        x = self.featex02(x)    
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def representation(self, x):                
        x = self.conv01(x)
        x = self.pool01(x)
        x = self.featex01(x)
        x = self.featex02(x) 
        x = x.view(x.size(0), -1)
        return x



class FeatExp(nn.Module):  

    def __init__(self, in_channels, ):
        super(FeatExp, self).__init__()
        self.conv02b   = BasicConv2d(in_channels, 96, kernel_size=1, stride=1, padding=0)
        self.pool02a   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv02bb  = BasicConv2d(96, 208, kernel_size=3, stride=1, padding=1)
        self.conv02aa  = BasicConv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)        
        self.pool02    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        xb  =   self.conv02b(x)
        xa  =   self.pool02a(x)
        xbb =   self.conv02bb(xb)
        xaa =   self.conv02aa(xa)       
        xo  =   self.pool02( torch.cat([xbb, xaa], 1) )
        return xo


class BasicConv2d(nn.Module):    

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        nn.init.xavier_normal(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


