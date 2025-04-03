# import libraries
from torch.utils.data import random_split, DataLoader

# import custom modules
from dataset import torch, np, CustomDataset
from helpers import os, np, denormalize, nll_loss, new_model, load_model, test_model
from plot import plot_train_loss, plot_pull_histogram, plot_scatter

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

# paths to data
location = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # change here
DATA_DIR = os.path.join(location, '1_astronomy', 'data') # change here

MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

spectra_path = os.path.join(DATA_DIR, 'spectra.npy')
labels_path = os.path.join(DATA_DIR, 'labels.npy')
model_path = os.path.join(MODEL_DIR, 'model.pth')

# create a custom dataset
dataset = CustomDataset(spectra_path, labels_path)

# hyperparameters
batch_size = 32
learning_rate = 2e-4
num_epochs = 10

# experiment name
experiment_name = f"astronomy_bs{batch_size}_lr{learning_rate}_ep{num_epochs}"

# split Data
num_samples = len(dataset)
train_size = int(0.7 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# define criterion
criterion = nll_loss

# create new model or load existing model
if not os.path.exists(model_path):
    print("No model found, creating a new model...")
    model, train_losses, val_losses, writer = \
        new_model(
            experiment_name, 
            criterion, 
            device, 
            learning_rate, 
            train_loader, 
            val_loader, 
            num_epochs, 
            model_path
        )
    # plot model losses
    plot_train_loss(
        train_losses,
        val_losses,
        experiment_name,
    )

else:
    new_model_input = input("A model exists, do you want to train a new the model? (y/n): ").strip().lower()

    if new_model_input == 'y':
        print("Preparing to train a new model...")
        model, train_losses, val_losses, writer = \
            new_model(
                experiment_name,
                criterion, 
                device, 
                learning_rate, 
                train_loader, 
                val_loader, 
                num_epochs, 
                model_path
            )
        # plot model losses
        plot_train_loss(
            train_losses,
            val_losses,
            experiment_name,
        )

    # if user does not want to create a new model, load the existing model
    else:
        print("Preparing to load the model...")
        model = load_model(model_path, device)

# test the model
test_predictions, true_labels = test_model(
    model,
    test_loader,
    criterion,
    device,
)

# convert test_predictions and true_labels to numpy arrays so they can be sliced
test_predictions = np.concatenate(test_predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)

# extract predicted means and log standard deviations from test_predictions
predicted_means = test_predictions[:, :3]
log_stds = test_predictions[:, 3:]

# convert log standard deviations to standard deviations
predicted_stds = np.exp(log_stds)

# plot pull histogram
plot_pull_histogram(
    true_labels,
    predicted_means,
    predicted_stds,
    dataset.num_labels,
    dataset.name_labels,
    experiment_name,
)

# denormalize true_labels and predicted_means
denormalized_true_labels = denormalize(true_labels, dataset.y_mean, dataset.y_std)
denormalized_predicted_means = denormalize(predicted_means, dataset.y_mean, dataset.y_std)

# plot scatter plot
plot_scatter(
    denormalized_true_labels,
    denormalized_predicted_means,
    dataset.num_labels,
    dataset.name_labels,
    experiment_name,
)