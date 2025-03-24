import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=8, num_classes=2, num_fc1=100, dropout_rate=0.25):
        
        super(CNNModel, self).__init__()
    
        self.dropout_rate = dropout_rate
        
        # Convolution Layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=2*out_channels, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=2*out_channels, out_channels=4*out_channels, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=4*out_channels, out_channels=8*out_channels, kernel_size=3, stride=1, padding=0)
        
        # compute the flatten size
        self.num_flatten = 8 * out_channels * 13 * 13
        
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, self.num_flatten)
        
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        
        return F.log_softmax(X, dim=1)