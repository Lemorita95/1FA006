import numpy as np
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim

from helpers import os, torch, MAIN_PATH, train_model
from model import nn, Generator, Discriminator


MODEL_DIR = os.path.join(MAIN_PATH, 'models')

# check for available devices and select if available
if torch.cuda.is_available():
    device = torch.device("cuda")       #CUDA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")        #Apple GPU
else:
    device = torch.device("cpu")        #if nothing is found use the CPU
print(f"Using device: {device}")

# use random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# hyperparameters
batch_size = 64
learning_rate = 2e-4
num_epochs = 50
patience = 10
logStep = 625  # the number of steps to log the images and losses to tensorboard

latent_dimension = 128 # 64, 128, 256
# for simplicity we will flatten the image to a vector and to use simple MLP networks
# 28 * 28 * 1 flattens to 784
# you are also free to use CNNs
image_dimension = 28 * 28 * 1  # 784
channels_img = 1

# we define a tranform that converts the image to tensor and normalizes it with mean and std of 0.5
# which will convert the image range from [0, 1] to [-1, 1]
myTransforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
dataset = datasets.MNIST(root="dataset/", transform=myTransforms, download=True)
# let's create a dataloader to load the data in batches
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# initialize networks and optimizers
discriminator = Discriminator(channels_img=channels_img).to(device)
generator = Generator(latent_dimension, channels_img=channels_img).to(device)
opt_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# This is a binary classification task, so we use Binary Cross Entropy Loss
criterion = nn.BCELoss()

train_model(num_epochs, loader, device, latent_dimension, generator, discriminator, opt_generator, opt_discriminator, criterion, logStep)

# Save the model
print("Model trained, saving...")
generator_path = os.path.join(MODEL_DIR, 'generator.pth')
discriminator_path = os.path.join(MODEL_DIR, 'discriminator.pth')

os.makedirs(os.path.dirname(generator_path), exist_ok=True)

torch.save(generator.state_dict(), "generator.pth")
print(f"Model saved to {generator_path}")
torch.save(discriminator.state_dict(), "discriminator.pth")
print(f"Model saved to {discriminator_path}")
