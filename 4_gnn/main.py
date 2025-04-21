import os
import awkward
from torch.utils.data import DataLoader

# local modules
from helpers import MAIN_PATH, torch, get_normalized_data, denormalize, collate_fn_gnn, new_model, load_model, test_model
from model import GNNEncoder
from plot import np, plot_train_loss, plot_scatter, plot_residuals_scatter, plot_residuals_distribution

DATA_PATH = os.path.join(MAIN_PATH, 'data') # change here
MODEL_DIR = os.path.join(MAIN_PATH, 'models')

# Set the device to GPU if available, otherwise use CPU
# will not use mps due to unsupported operations of torch_geometric
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# use random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# hyperparameters
batch_size = 64
learning_rate = 0.8e-4
num_epochs = 50
patience = 10

# Load the dataset
train_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "train.pq"))
val_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "val.pq"))
test_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "test.pq"))

# Normalize data and labels
features_mean, features_std, labels_mean, labels_std = get_normalized_data(train_dataset, val_dataset, test_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_gnn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn)

# Initialize the GNNEncoder, at least 2 dynamic edge conv layers
n_features = 3 # (t, x, y)
hidden_dim = 128
output_dim = 2 # (xpos, ypos)
k_neighbors = 10
n_layers = 2
model = GNNEncoder(n_features=n_features, hidden_dim=hidden_dim, output_dim=output_dim, k_neighbors=k_neighbors, num_layers=n_layers)

criterion = torch.nn.MSELoss()

# experiment name and model name
experiment_name = f"gnn_bs{batch_size}_lr{learning_rate}_ep{num_epochs}_l{n_layers}_hd{hidden_dim}_k{k_neighbors}"
model_name = f'model_l{n_layers}_hd{hidden_dim}_k{k_neighbors}'
model_path = os.path.join(MODEL_DIR, f'{model_name}.pth')
print(f"Experiment name: {experiment_name}")
print(f"Model name: {model_name}\n")

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
            model_path,
            patience
        )
    # plot model losses
    plot_train_loss(
        train_losses,
        val_losses,
        experiment_name,
    )

else:
    new_model_input = input("A model exists, do you want to train a new the model? (y/n): ").strip().lower()
    print()

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
                model_path,
                patience
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

# denormalize the predictions and true labels
test_predictions = denormalize(test_predictions, labels_mean, labels_std)
true_labels = denormalize(true_labels, labels_mean, labels_std)

# plot scatter plot
plot_scatter(
    true_labels,
    test_predictions,
    output_dim,
    ['xpos', 'ypos'],
    ['m', 'm'],
    experiment_name,
)

# plot residuals
plot_residuals_scatter(
    true_labels,
    test_predictions,
    output_dim,
    ['xpos', 'ypos'],
    ['m', 'm'],
    experiment_name,
)

# plot residuals
plot_residuals_distribution(
    true_labels,
    test_predictions,
    output_dim,
    ['xpos', 'ypos'],
    ['m', 'm'],
    experiment_name,
)