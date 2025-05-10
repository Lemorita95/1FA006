# importing necessary libraries
import matplotlib.pyplot as plt
from torchvision import utils, transforms # For image transforms

from helpers import os, time

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


def generate_plot_samples(diffusion, num_samples):
    """
    Generates and saves a grid of sampled images using a diffusion model.
    Args:
        diffusion: The diffusion model instance used to generate samples. 
                   It should have a `sample` method for the backward pass.
        num_samples (int): The number of samples to generate.
    Returns:
        None. The function saves the generated image grid to a file and displays it.
    Side Effects:
        - Saves the generated image grid as "sampled_images.png" in the directory specified by `IMAGES_DIR`.
        - Opens and displays the generated image grid using the default image viewer.
    """

    # you can obtain sampled images (i.e. the backward pass) by calling the sample function
    sampling_start_time = time.time()
    sampled_images = diffusion.sample(batch_size = num_samples)
    print(f"Generated {num_samples} samples in {time.time() - sampling_start_time:.2f} seconds ({(time.time() - sampling_start_time)/num_samples:.2f} seconds/sample)")

    samples = sampled_images.reshape(-1, 1, 28, 28)
    samples_grid = utils.make_grid(samples, normalize=True)

    img = transforms.ToPILImage()(samples_grid)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    img.save(os.path.join(IMAGES_DIR, "sampled_images.png"))
    img.show()