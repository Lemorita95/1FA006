# load libraries
import torch.nn as nn

class CNNModel(nn.Module):
    '''
    object that represents the cnn model
    '''
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # first conv-maxpool layer
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        # second conv-maxpool layer
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        # third conv-maxpool layer
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        # fully connected hidden layer
        '''
        64 is the number of filters in self.conv3, the last conv layer
        16384 is the length of the input signal
        2^3 is the reduction factor after 3 maxpool layers
        '''
        self.layer4 = nn.Linear(64 * (16384 // (2 ** 3)), 128)
        self.relu4 = nn.ReLU()

        # output layer
        self.output_layer = nn.Linear(128, 3)

    def forward(self, x):
        '''
        forward pass of the model
        '''
        # first conv-maxpool layer
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)

        # second conv-maxpool layer
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)

        # third conv-maxpool layer
        x = self.relu3(self.conv3(x))
        x = self.maxpool3(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fully connected hidden layer
        x = self.relu4(self.layer4(x))

        # output layer
        x = self.output_layer(x)

        return x