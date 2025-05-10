import torch.nn as nn

class DiffusionModel(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model implemented using PyTorch's nn.Module.
        input data: current sample value and the time step
        output data: amount of noise added in timestep t
    Attributes:
        layer (nn.Linear): The layer of the MLP.
        activtion (nn.ReLU): The activation function applied after the layer.
    Methods:
        forward(x):
            Performs a forward pass through the MLP model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 2).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, 1).
    """
    def __init__(self):
        super(DiffusionModel, self).__init__()

        # Define the MLP architecture
        self.layer1 = nn.Linear(2, 16)
        self.activtion1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 16)
        self.activtion2 = nn.ReLU()
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        # Forward pass through the model
        x = self.layer1(x)
        x = self.activtion1(x)

        x = self.layer2(x)
        x = self.activtion2(x)

        x = self.output(x)

        return x