import torch.nn as nn

class Generator(nn.Module):
    """
    https://arxiv.org/pdf/1511.06434
    """
    def __init__(self, latent_dimension, feature_maps_gen=128, channels_img=1):
        super(Generator, self).__init__()
        
        # input layer: latent_dimension -> (feature_maps_gen * 4) x 7 x 7
        self.input_layer = nn.ConvTranspose2d(latent_dimension, feature_maps_gen * 4, kernel_size=7, stride=1, padding=0, bias=False)
        self.input_norm = nn.BatchNorm2d(feature_maps_gen * 4)
        self.input_actv = nn.ReLU()

        # convolution layer 1: (feature_maps_gen * 4) x 7 x 7 -> (feature_maps_gen * 2) x 14 x 14
        self.conv1_layer = nn.ConvTranspose2d(feature_maps_gen * 4, feature_maps_gen * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv1_norm = nn.BatchNorm2d(feature_maps_gen * 2)
        self.conv1_actv = nn.ReLU()

        # output layer: (feature_maps_gen * 2) x 14 x 14 -> channels_img x 28 x 28
        self.output_layer = nn.ConvTranspose2d(feature_maps_gen * 2, channels_img, kernel_size=4, stride=2, padding=1, bias=False)
        self.output_actv = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = self.input_actv(x)

        x = self.conv1_layer(x)
        x = self.conv1_norm(x)
        x = self.conv1_actv(x)

        x = self.output_layer(x)
        x = self.output_actv(x)

        return x


class Discriminator(nn.Module):
    """
    https://arxiv.org/pdf/1511.06434
    """
    def __init__(self, channels_img=1, feature_maps_disc=64):  # Grayscale images
        super(Discriminator, self).__init__()
        
        # input layer: channels_img x 28 x 28 -> (feature_maps_disc) x 14 x 14
        self.input_layer = nn.Conv2d(channels_img, feature_maps_disc, kernel_size=4, stride=2, padding=1, bias=False)
        self.input_actv = nn.LeakyReLU(0.2)

        # convolution layer 1: (feature_maps_disc) x 14 x 14 -> (feature_maps_disc * 2) x 7 x 7
        self.conv1_layer = nn.Conv2d(feature_maps_disc, feature_maps_disc * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv1_norm = nn.BatchNorm2d(feature_maps_disc * 2)
        self.conv1_actv = nn.LeakyReLU(0.2)

        # output layer: (feature_maps_disc * 2) x 7 x 7 -> 1 x 1 x 1
        self.output_layer = nn.Conv2d(feature_maps_disc * 2, 1, kernel_size=7, stride=1, padding=0, bias=False)
        self.output_actv = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_actv(x)

        x = self.conv1_layer(x)
        x = self.conv1_norm(x)
        x = self.conv1_actv(x)

        x = self.output_layer(x)
        x = self.output_actv(x)

        # flatten
        x = x.view(-1, 1).squeeze(1)

        return x