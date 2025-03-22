import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lsq import Conv2dLSQ, LinearLSQ

def findConv2dOutShape(hin,win,conv,pool=2):
    # get conv arguments
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)

# Neural Network
class CNNModel_Q(nn.Module):
    
    # Network Initialisation
    def __init__(self, params):
        
        super(CNNModel_Q, self).__init__()
    
        Cin,Hin,Win=params["shape_in"]
        init_f=params["initial_filters"] 
        num_fc1=params["num_fc1"]  
        num_classes=params["num_classes"] 
        self.dropout_rate=params["dropout_rate"] 
        nbits = params['nbits']
        
        # Convolution Layers
        self.conv1 = Conv2dLSQ(Cin, init_f, kernel_size=3, nbits=nbits)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = Conv2dLSQ(init_f, 2*init_f, kernel_size=3, nbits=nbits)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = Conv2dLSQ(2*init_f, 4*init_f, kernel_size=3, nbits=nbits)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = Conv2dLSQ(4*init_f, 8*init_f, kernel_size=3, nbits=nbits)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = LinearLSQ(self.num_flatten, num_fc1, nbits=nbits)
        self.fc2 = LinearLSQ(num_fc1, num_classes, nbits=nbits)

    def forward(self,X):
        
        # Convolution & Pool Layers
        X = F.relu(self.conv1(X)); 
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, self.num_flatten)
        
        X = F.relu(self.fc1(X))
        X=F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)