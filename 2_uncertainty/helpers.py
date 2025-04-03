import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import CNNModel

def normalize(y):
    """
    Normalizes the input array by subtracting the mean and dividing by the standard deviation.

    Parameters:
    y (numpy.ndarray): Input array to be normalized.

    Returns:
    tuple: A tuple containing:
        - normalized_y (numpy.ndarray): The normalized array.
        - mean (numpy.ndarray): The mean of the input array along the specified axis.
        - std (numpy.ndarray): The standard deviation of the input array along the specified axis.
    """

    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    normalized_y = (y - mean) / std

    return normalized_y, mean, std


def denormalize(y, mean, std):
    """
    Denormalizes the input array by multiplying by the standard deviation and adding the mean.

    Parameters:
    y (numpy.ndarray): Input array to be denormalized.
    mean (numpy.ndarray): The mean used for normalization.
    std (numpy.ndarray): The standard deviation used for normalization.

    Returns:
    numpy.ndarray: The denormalized array.
    """

    return y * std + mean


def nll_loss(predictions, batch_labels, n_labels):
    """
    Computes the Negative Log-Likelihood (NLL) loss for a given set of predictions and labels.
    This function assumes that the predictions contain both the mean and log standard deviation
    of a Gaussian distribution for each label. The loss is computed based on the 
    Gaussian NLL formula.
    Args:
        predictions (torch.Tensor): A tensor of shape (batch_size, 2 * n_labels) containing
            the predicted mean and log standard deviation for each label. The first `n_labels` columns
            correspond to the mean values, and the remaining `n_labels` columns correspond
            to the log standard deviation values.
        batch_labels (torch.Tensor): A tensor of shape (batch_size, n_labels) containing
            the ground truth labels for the batch.
        n_labels (int): The number of labels being predicted.
    Returns:
        torch.Tensor: A scalar tensor representing the mean NLL loss over the batch.
    """

    # extract mean value from cnn output
    mean = predictions[:, :n_labels]

    # extract log standard deviation from cnn output
    log_std = predictions[:, n_labels:]

    # converte log standard deviation to standard deviation
    std = torch.exp(log_std)

    # return loss based on the Gaussian NLL formula
    return torch.mean((0.5 * ((batch_labels - mean) / std) ** 2) + log_std)


def train_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, writer):
    """
    Trains and validates a PyTorch model for a specified number of epochs.
    Args:
        model (torch.nn.Module): The PyTorch model to be trained and validated.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (callable): Loss function to calculate the error between predictions and targets.
        optimizer (torch.optim.Optimizer): Optimizer to update model parameters.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to run the model and data on (e.g., 'cpu' or 'cuda').
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging metrics and histograms.
    Returns:
        tuple: A tuple containing:
            - train_losses (list): List of average training losses for each epoch.
            - val_losses (list): List of average validation losses for each epoch.
            - model (torch.nn.Module): The trained model.
            - writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer with logged data.
    Notes:
        - The function alternates between training and validation phases for each epoch.
        - During training, the model is set to training mode to enable batch normalization and dropout.
        - During validation, the model is set to evaluation mode to disable batch normalization and dropout.
        - Training and validation losses are logged to TensorBoard for visualization.
        - Model parameter histograms are also logged to TensorBoard for each epoch.
    """

    # empty list to store the training and validation losses
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
            loss = criterion(predictions, batch_y, batch_y.shape[1]) # calculate the loss based on the prediction

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
                loss = criterion(predictions, batch_y, batch_y.shape[1]) # calculate the loss based on the prediction
                val_loss += loss.item() # calulate loss per batch

        val_loss /= len(val_loader) # calulate loss per epoch
        val_losses.append(val_loss)

        writer.add_scalar("val_loss", val_loss, epoch)

        # Print progress (print every epoch)
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

    return train_losses, val_losses, model, writer


def test_model(model, test_loader, criterion, device):
    """
    Evaluates a trained model on a test dataset and computes the loss and predictions.
    Args:
        model (torch.nn.Module): The trained PyTorch model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function used to compute the loss.
        device (torch.device): Device on which the computation will be performed (e.g., 'cpu' or 'cuda').
    Returns:
        tuple: A tuple containing:
            - test_predictions (list): List of numpy arrays containing the model's predictions for each batch.
            - true_labels (list): List of numpy arrays containing the true labels for each batch.
    Notes:
        - The model is set to evaluation mode using `model.eval()` to disable dropout and batch normalization.
        - Gradients are not computed during evaluation by using `torch.no_grad()`.
        - The input tensors are moved to the specified device, and an additional channel dimension is added for Conv1d compatibility.
        - The function prints the final test loss averaged over all batches.
    """

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
            loss = criterion(predictions, batch_y, batch_y.shape[1]) # calculate the loss based on the prediction
            test_loss += loss.item() # calulate loss per batch

            # Append predictions and true labels for plotting
            test_predictions.append(predictions.cpu().numpy())
            true_labels.append(batch_y.cpu().numpy())

    test_loss /= len(test_loader) # calculate total loss
    print(f"Final Test Loss: {test_loss:.4f}")

    return test_predictions, true_labels


def new_model(experiment_name, criterion, device, learning_rate, train_loader, val_loader, num_epochs, model_path):
        #Initialize the model
        model = CNNModel().to(device)
        criterion = nll_loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # define writer for tensorboard logging
        writer = SummaryWriter(f"runs/{experiment_name}")

        # log the network architecture
        dummy_input = torch.randn(1, 1, 16384).to(device)
        writer.add_graph(model, dummy_input)

        # Train the model
        train_losses, val_losses, model, writer = train_validate_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs,
            device,
            writer,
        )

        #close the tensorboard writer
        writer.flush()
        writer.close()

        # Save the model
        print("Model trained, saving...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        return model, train_losses, val_losses, writer


def load_model(model_path, device):
    # Load the model
    print("Loading existing model...")
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")

    return model