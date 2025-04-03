# load libraries
import torch.nn as nn

class CNNModel(nn.Module):
    """
    CNNModel is a convolutional neural network (CNN) implemented using PyTorch's `nn.Module`. 
    This model is designed for 1D input data and consists of multiple convolutional, activation, 
    and max-pooling layers, followed by fully connected layers for classification or regression tasks.
    Attributes:
        conv-maxpool layers: 5 layers with ReLU activation functions.
        linear (hidden) layers: 2 layers with Softplus activation functions.
        output layer: 1 layer with 6 output features.
    Methods:
        forward(x):
            Defines the forward pass of the model. Takes a 1D input tensor `x` and passes it through 
            the convolutional, activation, max-pooling, and fully connected layers sequentially. 
            Returns the output tensor.
    """

    def __init__(self):
        super(CNNModel, self).__init__()
        
        # first conv-maxpool layer
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.atv_conv1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        # second conv-maxpool layer
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.atv_conv2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        # third conv-maxpool layer
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.atv_conv3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        # fourth conv-maxpool layer
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.atv_conv4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)

        # fifth conv-maxpool layer
        self.conv5 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.atv_conv5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool1d(kernel_size=2)

        # fully connected hidden layer, data size reduced by 2^3 after 3 maxpool layers
        self.linear1 = nn.Linear(64 * (16384 // (2 ** 5)), 128)
        self.atv_linear1 = nn.Softplus()

        # fully connected hidden layer, data size reduced by 2^3 after 3 maxpool layers
        self.linear2 = nn.Linear(128, 64)
        self.atv_linear2 = nn.Softplus()

        # output layer
        self.output_layer = nn.Linear(64, 6)

    def forward(self, x):
        # first conv-maxpool layer
        x = self.atv_conv1(self.conv1(x))
        x = self.maxpool1(x)

        # second conv-maxpool layer
        x = self.atv_conv2(self.conv2(x))
        x = self.maxpool2(x)

        # third conv-maxpool layer
        x = self.atv_conv3(self.conv3(x))
        x = self.maxpool3(x)

        # third conv-maxpool layer
        x = self.atv_conv4(self.conv4(x))
        x = self.maxpool4(x)

        # third conv-maxpool layer
        x = self.atv_conv5(self.conv5(x))
        x = self.maxpool5(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fully connected hidden layer
        x = self.atv_linear1(self.linear1(x))
        x = self.atv_linear2(self.linear2(x))

        # output layer
        x = self.output_layer(x)

        # return value
        return x