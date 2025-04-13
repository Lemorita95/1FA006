# import libraries
from torch.utils.data import random_split, DataLoader

# import custom modules
from model import CNNModel, TinyCNN, CombinedModel
from dataset import torch, np, CustomDataset
from helpers import os, np, nf_loss, new_model, load_model, test_model, denormalize
from plot import plot_train_loss, plot_pull_histogram, plot_scatter, plot_residuals

# check for available devices and select if available
if torch.cuda.is_available():
    device = torch.device("cuda")       #CUDA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")        #Apple GPU
else:
    device = torch.device("cpu")        #if nothing is found use the CPU
print(f"Using device: {device}")

# handle mps double precision
flow_types = ["diagonal_gaussian", "full_gaussian", "full_flow"]
flow_type = flow_types[2] # change here
if flow_type == "full_flow" and device.type == "mps":
    # MPS does not support double precision, therefore we need to run the flow on the CPU
    fp64_on_cpu = True
else:
    fp64_on_cpu = False

# use random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# paths
location = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # change here
DATA_DIR = os.path.join(location, '1_astronomy', 'data') # change here

spectra_path = os.path.join(DATA_DIR, 'spectra.npy')
labels_path = os.path.join(DATA_DIR, 'labels.npy')

MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

# hyperparameters
batch_size = 64
learning_rate = 0.8e-4
num_epochs = 10

# define what model to use
cnn_model = TinyCNN

# experiment name and model name
experiment_name = f"astronomy_bs{batch_size}_lr{learning_rate}_ep{num_epochs}_{flow_type}_{cnn_model.__name__}"
model_path = os.path.join(MODEL_DIR, f'model_{flow_type}_{cnn_model.__name__}.pth')

# create a custom dataset
dataset = CustomDataset(spectra_path, labels_path)

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

# define combined model and criterion
model = CombinedModel(cnn_model, nf_type=flow_type, fp64_on_cpu=fp64_on_cpu).to(device)
criterion = nf_loss

# create new model or load existing model
if not os.path.exists(model_path):
    print("No model found, creating a new model...")
    train_losses, val_losses = \
        new_model(
            experiment_name, 
            model,
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
        train_losses, val_losses = \
            new_model(
                experiment_name,
                model,
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
        load_model(model_path, model)

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
predicted_std = test_predictions[:, 3:]

# Combined model already predicts the standard deviation
predicted_stds = predicted_std

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
    dataset.unit_labels,
    experiment_name,
)

plot_residuals(
    denormalized_true_labels,
    denormalized_predicted_means,
    dataset.num_labels,
    dataset.name_labels,
    dataset.unit_labels,
    experiment_name,
)
