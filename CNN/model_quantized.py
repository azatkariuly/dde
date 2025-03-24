import torch.nn as nn
import torch.nn.functional as F
from lsq import Conv2dLSQ, LinearLSQ, ActLSQ

class CNNModel_LSQ(nn.Module):
    def __init__(self, in_channels=3, out_channels=8, num_classes=2, num_fc1=100, dropout_rate=0.25, nbits=8):
        
        super(CNNModel_LSQ, self).__init__()
    
        self.dropout_rate = dropout_rate
        self.nbits = nbits
        
        # Convolution Layers
        self.conv1 = Conv2dLSQ(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, nbits=self.nbits)
        self.act1 = ActLSQ(nbits=self.nbits)
        self.conv2 = Conv2dLSQ(in_channels=out_channels, out_channels=2*out_channels, kernel_size=3, stride=1, padding=0, nbits=self.nbits)
        self.act2 = ActLSQ(nbits=self.nbits)
        self.conv3 = Conv2dLSQ(in_channels=2*out_channels, out_channels=4*out_channels, kernel_size=3, stride=1, padding=0, nbits=self.nbits)
        self.act3 = ActLSQ(nbits=self.nbits)
        self.conv4 = Conv2dLSQ(in_channels=4*out_channels, out_channels=8*out_channels, kernel_size=3, stride=1, padding=0, nbits=self.nbits)
        self.act4 = ActLSQ(nbits=self.nbits)
        
        # compute the flatten size
        self.num_flatten = 8 * out_channels * 13 * 13
        
        self.fc1 = LinearLSQ(self.num_flatten, num_fc1, nbits=self.nbits)
        self.act5 = ActLSQ(nbits=self.nbits)
        self.fc2 = LinearLSQ(num_fc1, num_classes, nbits=self.nbits)

    def forward(self, X):
        X = self.act1(F.relu(self.conv1(X)))
        X = F.max_pool2d(X, 2, 2)
        X = self.act2(F.relu(self.conv2(X)))
        X = F.max_pool2d(X, 2, 2)
        X = self.act3(F.relu(self.conv3(X)))
        X = F.max_pool2d(X, 2, 2)
        X = self.act4(F.relu(self.conv4(X)))
        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, self.num_flatten)
        
        X = self.act5(F.relu(self.fc1(X)))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        
        return F.log_softmax(X, dim=1)