# importing necessary libraries
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import r2_score 

from helpers import np, os

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

def plot_pull_histogram(true_labels, predicted_means, predicted_stds, num_labels, name_labels, experiment_name):
    """
    Plots pull histograms for multiple labels, showing the distribution of 
    normalized residuals (pulls) and fitting a normal distribution to each.
    Parameters:
    -----------
    true_labels : numpy.ndarray
        Array of true labels with shape (n_samples, n_labels).
    predicted_means : numpy.ndarray
        Array of predicted means with shape (n_samples, n_labels).
    predicted_stds : numpy.ndarray
        Array of predicted standard deviations with shape (n_samples, n_labels).
    num_labels : int
        Number of labels to plot.
    name_labels : list of str
        List of label names corresponding to each label.
    experiment_name : str
        Name of the experiment, used for saving the plot file.
    Returns:
    --------
    None
        The function saves the plot as an image file and displays it.
    Notes:
    ------
    - The pull is calculated as: (true_labels - predicted_means) / predicted_stds.
    - Each subplot shows the histogram of pull values for a label, overlaid with 
      a fitted normal distribution.
    - The mean and standard deviation of the fitted normal distribution are 
      annotated on each subplot.
    - The plot is saved as a PNG file in the directory specified by `IMAGES_DIR` 
      with the filename format: `pull_distribution_<experiment_name>.png`.
    """

    # compute the pull for each label
    pulls = (true_labels - predicted_means) / predicted_stds

    # plot the pull distribution for each label
    plt.figure(figsize=(15, 5))

    for i in range(num_labels):
        plt.subplot(1, num_labels, i + 1)
        
        # fit a normal distribution to the pull values
        mu, sigma = norm.fit(pulls[:, i])
        
        # plot the histogram of the pull values
        n, bins, _ = plt.hist(pulls[:, i], bins=30, density=True, alpha=0.7, color='gray', edgecolor='black', label="Pull Histogram")
        
        # plot the fitted normal distribution
        x = np.linspace(bins[0], bins[-1], 100)
        pdf = norm.pdf(x, mu, sigma)
        plt.plot(x, pdf, 'r-', label="Normal fit")

        # annotate the mean and standard deviation
        plt.annotate(
        f"Mean: {mu:.2f}\nSTD: {sigma:.2f}",
        xy=(1, 1),         # Annotate near the upper-right corner
        xycoords='axes fraction',  # Use fraction of axes (0 to 1)
        xytext=(.95, .95), # Position the annotation box
        textcoords='axes fraction', # Use fraction for the text position
        ha='right',        # Horizontal alignment
        va='top',          # Vertical alignment
        fontsize=10,       # Font size
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.25')  # Box around text
    )
        
        # axis configurations
        plt.xlabel("(Predicted - True) / Uncertainty")
        plt.ylabel("Counts")
        plt.title(f"{name_labels[i]}")
        plt.legend(loc='upper left')

    plt.tight_layout()
    filename = f"pull_distribution_{experiment_name}.png"
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMAGES_DIR, filename))
    plt.show()

def plot_scatter(true_labels, predicted_means, num_labels, name_labels, unit_labels, experiment_name):
    

    plt.figure(figsize=(15, 5))

    for i in range(num_labels):

        # calculate r2 score
        r2 = r2_score(true_labels[:, i], predicted_means[:, i])

        plt.subplot(1, num_labels, i + 1)  # Create a subplot for each label
        plt.scatter(true_labels[:, i], predicted_means[:, i], alpha=0.5, label=name_labels[i])
        plt.plot([true_labels[:, i].min(), true_labels[:, i].max()],
                [true_labels[:, i].min(), true_labels[:, i].max()],
                color="red", linestyle="--", label="True values line")
        plt.xlabel(f"True Values [{unit_labels[i]}]")
        plt.ylabel(f"Test Predictions [{unit_labels[i]}]")
        plt.title(f"{name_labels[i]} (R2 = {r2:.2f})")
        plt.legend()

    plt.tight_layout()
    filename = f"predictions_{experiment_name}.png"
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMAGES_DIR, filename))
    plt.show()


def plot_residuals(true_labels, predicted_means, predicted_std, num_labels, name_labels, unit_labels, experiment_name):
    
    plt.figure(figsize=(15, 5))

    for i in range(num_labels):

        plt.subplot(1, num_labels, i + 1)  # Create a subplot for each label
        plt.scatter(true_labels[:, i], true_labels[:, i] - predicted_means[:, i], alpha=0.5, label=name_labels[i])
        plt.hlines(0, true_labels[:, i].min(), true_labels[:, i].max(), color="red", linestyle="--", label="Zero line")
        plt.xlabel(f"True Values [{unit_labels[i]}]")
        plt.ylabel(f"Residuals (True - Predicted) [{unit_labels[i]}]")
        plt.title(f"{name_labels[i]}")
        plt.legend()

    plt.tight_layout()
    filename = f"residuals_{experiment_name}.png"
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMAGES_DIR, filename))
    plt.show()
