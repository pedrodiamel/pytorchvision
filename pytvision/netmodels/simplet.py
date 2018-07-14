#----------------------------------------------------------------------------------------------
# SimpleNet
# Pedro D. Marrero Fernandez
#----------------------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SimpleNet', 'simplenet']

def simplenet(pretrained=False, **kwargs):
    r"""Simple model architecture
    """
    model = SimpleNet(**kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['simplenet']))
    return model


class SimpleNet(nn.Module):
    """
    Simple Convolutional Net
    """
    def __init__(self, num_channels=1, num_classes=8, size_input=64):
        super(SimpleNet, self).__init__()
        
        self.num_classes = num_classes        
        self.num_channels = num_channels
        self.size_input = size_input

        f=5; s=1; p=0
        n0 = self.size_input
        n1 = int((n0-2*p-f)/s+1)//2
        n2 = int((n1-2*p-f)/s+1)//2
        n3 = n2*n2*20
        self.dim = n3

        self.bn0 = nn.BatchNorm2d( num_channels )
        self.conv1 = nn.Conv2d( num_channels , 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d(p=0.3)
        self.fc1 = nn.Linear(n3, 512)
        self.fc2 = nn.Linear(512, num_classes)

        nn.init.xavier_normal(self.conv1.weight)
        nn.init.xavier_normal(self.conv2.weight)
        nn.init.xavier_normal(self.fc1.weight)
        nn.init.xavier_normal(self.fc2.weight)

    def forward(self, x):  

        x = self.bn0(x)
        x = F.relu(F.max_pool2d( self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), 2))
        x = x.view(x.size(0), -1)        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)        
        return x

    def representation(self, x):    
                    
        x = self.bn0(x)
        x = F.relu(F.max_pool2d( self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), 2))
        x = x.view(x.size(0), -1)
        return x