import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from helpers import os, MAIN_PATH
from plot import transforms


def get_dataloader(batch_size):
    # we define a tranform that converts the image to tensor
    myTransforms = transforms.Compose([transforms.ToTensor()])

    # the MNIST dataset is available through torchvision.datasets
    print("loading MNIST digits dataset")
    DATA_PATH = os.path.join(MAIN_PATH, 'dataset')

    mnist_train = datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=myTransforms)
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)

    mnist_val = datasets.MNIST(root=DATA_PATH, train=False, download=False, transform=myTransforms)
    val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader