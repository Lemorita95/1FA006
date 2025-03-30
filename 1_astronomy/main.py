# import libraries
import os
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score 


# import custom modules
from dataset import torch, np, CustomDataset
from model import CNNModel, nn

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
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
spectra_path = os.path.join(data_dir, 'spectra.npy')
labels_path = os.path.join(data_dir, 'labels.npy') # "mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h", "SNR"

# Create a custom dataset
dataset = CustomDataset(spectra_path, labels_path)
# dataset.to_device(device) # create tensors from np array and send to device

# # plot first 3 stars
# for i in range(3):
#   fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#   ax.plot(dataset.X[i], lw=1)
#   ax.set_title(f"Star {i}")
#   plt.tight_layout()
#   plt.show()

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Experiment name
experiment_name = "astronomy_bs32_lr0.001_ep10"

# Split Data
num_samples = len(dataset)

train_size = int(0.7 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Initialize the model
model = CNNModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# define writer for tensorboard logging
writer = SummaryWriter(f"runs/{experiment_name}")

# log the network architecture
dummy_input = torch.randn(1, 1, 16384).to(device)
writer.add_graph(model, dummy_input)

# Train the model
train_losses, val_losses = [], []

for epoch in range(num_epochs): # loop through every epoch
    # Training
    model.train() # The model should be in training mode to use batch normalization and dropout
    train_loss = 0
    for batch_x, batch_y in train_loader: # loop through every batch

        # move tensors to device
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Add channel dimension for Conv1d
        batch_x = batch_x.unsqueeze(1)  
        
        # set the gradients to zero
        optimizer.zero_grad() 

        # Forward pass
        predictions = model(batch_x) # make a prediction with the current model, uses the forward() method
        loss = criterion(predictions, batch_y) # calculate the loss based on the prediction

        # Backward pass and optimization
        loss.backward() # calculated the gradiets for the given loss
        optimizer.step() # updates the weights and biases for the given gradients
        train_loss += loss.item() # calulate loss per batch

    train_loss /= len(train_loader) # calulate loss per epoch
    train_losses.append(train_loss)

    writer.add_scalar("train_loss", train_loss, epoch)

    # Validation
    model.eval() # The model should be in eval mode to not use batch normalization and dropout
    val_loss = 0
    with torch.no_grad(): # make sure the gradients are not changed in this step
        for batch_x, batch_y in val_loader:

            # move tensors to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Add channel dimension for Conv1d
            batch_x = batch_x.unsqueeze(1)  

            # forward pass
            predictions = model(batch_x) # make a prediction with the current model
            loss = criterion(predictions, batch_y) # calculate the loss based on the prediction
            val_loss += loss.item() # calulate loss per batch

    val_loss /= len(val_loader) # calulate loss per epoch
    val_losses.append(val_loss)

    writer.add_scalar("val_loss", val_loss, epoch)

    # Print progress (print every epoch)
    if epoch % 1 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

# Test the model
test_predictions, true_labels = [], []
model.eval() # The model should be in eval mode to not use batch normalization and dropout
test_loss = 0
with torch.no_grad(): # make sure the gradients are not changed in this step
    for batch_x, batch_y in test_loader:

        # move tensors to device
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Add channel dimension for Conv1d
        batch_x = batch_x.unsqueeze(1)  
        
        # forward pass
        predictions = model(batch_x) # make a prediction with the current model
        loss = criterion(predictions, batch_y) # calculate the loss based on the prediction
        test_loss += loss.item() # calulate loss per batch

        # Append predictions and true labels for plotting
        test_predictions.append(predictions.cpu().numpy())
        true_labels.append(batch_y.cpu().numpy())

test_loss /= len(test_loader) # calculate total loss
print(f"Final Test Loss: {test_loss:.4f}")

# Plot training/validation loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", color="blue")
plt.plot(val_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig(f"images/loss_{experiment_name}.png")
plt.show()

#close the tensorboard writer
writer.flush()
writer.close()

# Plot predictions vs true values for each label
num_labels = predictions.shape[1]
labels = ["Surface temperature [K]", "Surface gravity [-]", "Metallicity [-]"]
plt.figure(figsize=(15, 5))

# Convert predictions and true labels to numpy arrays
test_predictions = np.concatenate(test_predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)

# Denormalize the predictions and true labels
test_predictions = test_predictions * dataset.y_std + dataset.y_mean
true_labels = true_labels * dataset.y_std + dataset.y_mean

for i in range(num_labels):

    # calculate r2 score
    r2 = r2_score(true_labels[:, i], test_predictions[:, i])

    plt.subplot(1, num_labels, i + 1)  # Create a subplot for each label
    plt.scatter(true_labels[:, i], test_predictions[:, i], alpha=0.5, label=labels[i])
    plt.plot([true_labels[:, i].min(), true_labels[:, i].max()],
             [true_labels[:, i].min(), true_labels[:, i].max()],
             color="red", linestyle="--", label="True values line")
    plt.xlabel("True Values")
    plt.ylabel("Test Predictions")
    plt.title(f"{labels[i]} (R2 = {r2:.2f})")
    plt.legend()

plt.tight_layout()
plt.savefig(f"images/predictions_{experiment_name}.png")
plt.show()