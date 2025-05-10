# importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns  # a useful plotting library on top of matplotlib
import os
import numpy as np
import torch

IMAGES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')

def plot_train_loss(train_losses, val_losses, experiment_name):
    """
    Plots the training and validation loss over epochs and saves the plot as an image.
    Args:
        train_losses (list or array-like): A list or array containing the training loss values for each epoch.
        val_losses (list or array-like): A list or array containing the validation loss values for each epoch.
        experiment_name (str): A string representing the name of the experiment, used to name the saved plot file.
    Saves:
        A PNG image of the plot in the directory specified by the `IMAGES_DIR` constant, 
        with the filename formatted as "loss_<experiment_name>.png".
    Displays:
        The plot of training and validation loss.
    """
    
    # Plot training/validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()
    filename = f"loss_{experiment_name}.png"
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMAGES_DIR, filename))
    plt.show()


def plot_results(dataset, samples, experiment_name):
    # plot the samples
    fig, ax = plt.subplots(1, 1)

    # Flatten both and convert array to tensor
    tensor_flat = dataset.flatten()
    array_tensor = torch.from_numpy(samples).to(dataset.dtype).flatten()

    # Concatenate and compute max and min
    combined = torch.cat([tensor_flat, array_tensor])
    max_val = combined.max().item()
    min_val = combined.min().item()
    x_lim = max(abs(min_val), abs(max_val))

    bins = np.linspace(-x_lim, +x_lim, 50)
    sns.kdeplot(dataset, ax=ax, color='blue', label='True distribution', linewidth=2)
    sns.histplot(samples, ax=ax, bins=bins, color='red', label='Sampled distribution', stat='density')
    ax.legend()
    ax.set_xlabel('Sample value')
    ax.set_ylabel('Sample count')

    plt.tight_layout()
    filename = f"result_{experiment_name}.png"
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMAGES_DIR, filename))
    plt.show()