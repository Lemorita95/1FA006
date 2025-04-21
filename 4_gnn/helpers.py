import time
import torch
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import awkward as ak
from torch_geometric.data import Data, Batch

MAIN_PATH = os.path.dirname(os.path.realpath(__file__)) # change here


def normalize(awkward_array, mean=None, std=None):
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

    if mean is None or std is None:
        # Calculate mean and std if not provided
        mean = ak.mean(awkward_array)
        std = ak.std(awkward_array)

    # otherwise, use the provided mean and std
    normalized_array = (awkward_array - mean) / std

    return normalized_array, mean, std


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


def denormalize_std(y, std):
    """
    Denormalizes the input array by multiplying by the standard deviation.

    Parameters:
    y (numpy.ndarray): Input array to be denormalized.
    std (numpy.ndarray): The standard deviation used for normalization.

    Returns:
    numpy.ndarray: The denormalized array.
    """

    return y * std


def normalize_feature(dataset, means=None, stds=None):
    """
    Normalizes the features of a dataset along the time, x, and y dimensions.
    This function normalizes the "data" field of the input dataset by calculating
    or using provided means and standard deviations for the time, x, and y dimensions.
    The normalized data is concatenated back into the dataset in-place.
    Args:
        dataset (awkward): A dictionary-like structure containing the dataset. 
                        The "data" field is expected to be a 3D array of shape 
                        (n_events, n_features, n_hits), where n_features includes 
                        time, x, and y dimensions.
        means (tuple, optional): A tuple of precomputed means for the time, x, and y dimensions.
                                 Defaults to None, in which case the means are computed from the data.
        stds (tuple, optional): A tuple of precomputed standard deviations for the time, x, and y dimensions.
                                Defaults to None, in which case the standard deviations are computed from the data.
    Returns:
        tuple: A tuple containing:
            - means (tuple): The computed or provided means for the time, x, and y dimensions.
            - stds (tuple): The computed or provided standard deviations for the time, x, and y dimensions.
    Notes:
        - The time dimension is indexed as `[:, 0:1, :]` to preserve its dimensionality as (n_events, 1, n_hits).
        - The function modifies the "data" field of the input dataset in-place.
    """
    # provided means and stds
    if means is not None and stds is not None:
        mean_times, mean_x, mean_y = means
        std_times, std_x, std_y = stds
    # otherwise set means and stds to None and calculate it at normalize()
    else:
        mean_times, mean_x, mean_y = None, None, None
        std_times, std_x, std_y = None, None, None

    times = dataset["data"][:, 0:1, :]  # important to index the time dimension with 0:1 to keep this dimension (n_events, 1, n_hits)
                                            # with [:,0,:] we would get a 2D array of shape (n_events, n_hits)
    norm_times, mean_times, std_times = normalize(times, mean=mean_times, std=std_times)
    
    x = dataset["data"][:, 1:2, :]
    norm_x, mean_x, std_x = normalize(x, mean=mean_x, std=std_x)
    
    y = dataset["data"][:, 2:3, :]
    norm_y, mean_y, std_y = normalize(y, mean=mean_y, std=std_y)

    # inplace concatenate the normalized data back together
    dataset["data"] = ak.concatenate([norm_times, norm_x, norm_y], axis=1)

    # return means and stds for denormalization
    return tuple([mean_times, mean_x, mean_y]), tuple([std_times, std_x, std_y])


def normalize_label(dataset, means=None, stds=None):
    """
    Normalize the labels in the dataset (e.g., "xpos" and "ypos") using the provided means and standard deviations.
    If means and standard deviations are not provided, they will be calculated during normalization.
    Args:
        dataset (awkward): A dictionary-like structure containing the dataset with keys "xpos" and "ypos".
                           These keys should map to numerical data that will be normalized.
        means (tuple, optional): A tuple containing the mean values for "xpos" and "ypos".
                                 Defaults to None.
        stds (tuple, optional): A tuple containing the standard deviation values for "xpos" and "ypos".
                                Defaults to None.
    Returns:
        tuple: A tuple containing:
            - means (tuple): The mean values used for normalization (mean_xpos, mean_ypos).
            - stds (tuple): The standard deviation values used for normalization (std_xpos, std_ypos).
    """
    # provided means and stds
    if means is not None and stds is not None:
        mean_xpos, mean_ypos = means
        std_xpos, std_ypos = stds
    # otherwise set means and stds to None and calculate it at normalize()
    else:
        mean_xpos, mean_ypos = None, None
        std_xpos, std_ypos = None, None

    # Normalize labels (this can be done in-place), e.g. by
    norm_xpos, mean_xpos, std_xpos = normalize(dataset["xpos"], mean=mean_xpos, std=std_xpos)
    norm_ypos, mean_ypos, std_ypos = normalize(dataset["ypos"], mean=mean_ypos, std=std_ypos)

    # inplace assign the normalized labels back to the dataset
    dataset["xpos"] = norm_xpos
    dataset["ypos"] = norm_ypos
    
    # return means and stds for denormalization
    return tuple([mean_xpos, mean_ypos]), tuple([std_xpos, std_ypos])


def get_normalized_data(train_dataset, val_dataset, test_dataset):
    """
    Normalize the features and labels of training, validation, and test datasets.
    This function computes the mean and standard deviation of the features and labels 
    from the training dataset, then uses these statistics to normalize the features 
    and labels of the validation and test datasets. The normalization is performed 
    in-place, so the datasets are modified directly.
    Args:
        train_dataset: The dataset used for training. This dataset is used to compute 
                       the mean and standard deviation for normalization.
        val_dataset: The dataset used for validation. This dataset is normalized using 
                     the statistics computed from the training dataset.
        test_dataset: The dataset used for testing. This dataset is normalized using 
                      the statistics computed from the training dataset.
    Returns:
        tuple: A tuple containing four elements:
            - features_mean (float): The mean of the features computed from the training dataset.
            - features_std (float): The standard deviation of the features computed from the training dataset.
            - labels_mean (float): The mean of the labels computed from the training dataset.
            - labels_std (float): The standard deviation of the labels computed from the training dataset.
    """
    features_mean, features_std = normalize_feature(train_dataset)
    labels_mean, labels_std = normalize_label(train_dataset)

    # normalize the data and labels for validation - using mean and std from training
    # no need to save the mean and std, we already have them from train dataset
    _, _ = normalize_feature(val_dataset, features_mean, features_std)
    _, _ = normalize_label(val_dataset, labels_mean, labels_std)

    # normalize the data and labels for test - using mean and std from training
    # no need to save the mean and std, we already have them from traindataset
    _, _ = normalize_feature(test_dataset, features_mean, features_std)
    _, _ = normalize_label(test_dataset, labels_mean, labels_std)

    # as normalization occurs inplace, no need to return the datasets
    return features_mean, features_std, labels_mean, labels_std


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


def nf_loss(inputs, batch_labels, model):
    """
    Computes the loss for a normalizing flow model.

    Parameters
    ----------
    inputs : torch.Tensor
        The input data to the model.
    batch_labels : torch.Tensor
        The labels corresponding to the input data.
    model : torch.nn.Module
        The normalizing flow model used for evaluation.
    Returns
    -------
    torch.Tensor
        The computed loss value.
    """
    log_pdfs = model.log_pdf_evaluation(batch_labels, inputs) # get the probability of the labels given the input data
    loss = -log_pdfs.mean() # take the negative mean of the log probabilities
    return loss


def train_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, writer, patience):
    """
    Trains and validates a PyTorch model, with support for early stopping and TensorBoard logging.
    Args:
        model (torch.nn.Module): The PyTorch model to be trained and validated.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function to calculate the error.
        optimizer (torch.optim.Optimizer): Optimizer to update model parameters.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging metrics and histograms.
        patience (int): Number of epochs to wait for improvement in validation loss before early stopping.
    Returns:
        tuple: A tuple containing:
            - train_losses (list): List of training losses for each epoch.
            - val_losses (list): List of validation losses for each epoch.
            - model (torch.nn.Module): The trained model.
            - writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer used for logging.
    Notes:
        - The function logs training and validation losses to TensorBoard.
        - Early stopping is triggered if the validation loss does not improve for a specified number of epochs.
        - Histograms of model parameters are logged to TensorBoard at each epoch.
    """
    # empty list to store the training and validation losses
    train_losses, val_losses = [], []

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs): # loop through every epoch
        # Start timing the epoch
        epoch_start_time = time.time()

        # Training
        model.train() # The model should be in training mode to use batch normalization and dropout
        train_loss = 0
        for batch_x, batch_y in train_loader: # loop through every batch

            # move tensors to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
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

                # forward pass
                predictions = model(batch_x) # make a prediction with the current model
                loss = criterion(predictions, batch_y) # calculate the loss based on the prediction
                val_loss += loss.item() # calulate loss per batch

        val_loss /= len(val_loader) # calulate loss per epoch
        val_losses.append(val_loss)

        writer.add_scalar("val_loss", val_loss, epoch)

        # Print progress (print every epoch)
        epoch_duration = time.time() - epoch_start_time
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Duration = {epoch_duration:.2f}s")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"\tNo improvement in validation loss. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

    return train_losses, val_losses, model, writer, epoch


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
            
            # forward pass
            predictions = model(batch_x) # make a prediction with the current model
            loss = criterion(predictions, batch_y) # calculate the loss based on the prediction
            test_loss += loss.item() # calulate loss per batch

            # Append predictions and true labels for plotting
            test_predictions.append(predictions.cpu().numpy())
            true_labels.append(batch_y.cpu().numpy())

    test_loss /= len(test_loader) # calculate total loss
    print(f"Final Test Loss: {test_loss:.4f}")

    return test_predictions, true_labels


def new_model(experiment_name, model, criterion, device, learning_rate, train_loader, val_loader, num_epochs, model_path, patience=10):
    """
    Trains a given model, validates it, logs training progress to TensorBoard, and saves the trained model to a specified path.
    Args:
        experiment_name (str): Name of the experiment for TensorBoard logging.
        model (torch.nn.Module): The PyTorch model to be trained.
        criterion (torch.nn.Module): Loss function used for training.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        learning_rate (float): Learning rate for the optimizer.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        num_epochs (int): Number of epochs to train the model.
        model_path (str): File path to save the trained model.
    Returns:
        tuple: A tuple containing:
            - train_losses (list): List of training losses for each epoch.
            - val_losses (list): List of validation losses for each epoch.
    """
    #Initialize the model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # define writer for tensorboard logging
    writer = SummaryWriter(f"{os.path.join(MAIN_PATH, 'runs')}/{experiment_name}")

    # Train the model
    train_start_time = time.time()
    train_losses, val_losses, model, writer, epoch = train_validate_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs,
        device,
        writer,
        patience,
    )
    train_duration = time.time() - train_start_time
    print(f"\nTraining completed in {train_duration:.2f} seconds")
    print(f"Training time: {train_duration/epoch:.2f} seconds/epoch\n")

    #close the tensorboard writer
    writer.flush()
    writer.close()

    # Save the model
    print("Model trained, saving...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return train_losses, val_losses


def load_model(model_path, model):
    """
    Load the weights of a pre-trained model from a specified file path.
    Args:
        model_path (str): The file path to the saved model weights.
        model (torch.nn.Module): The PyTorch model instance to load the weights into.
    Returns:
        None
    """
    # Load the model
    print("Loading existing model...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Weights loaded to Model from {model_path}")


# Create the DataLoader for training, validation, and test datasets
# Important: We use the custom collate function to preprocess the data for GNN (see the description of the collate function for details)
def collate_fn_gnn(batch):
    """
    Custom function that defines how batches are formed.

    For a more complicated dataset with variable length per event and Graph Neural Networks,
    we need to define a custom collate function which is passed to the DataLoader.
    The default collate function in PyTorch Geometric is not suitable for this case.

    This function takes the Awkward arrays, converts them to PyTorch tensors,
    and then creates a PyTorch Geometric Data object for each event in the batch.

    You do not need to change this function.

    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list = []
    labels = []

    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the feautures are (time, x, y)
        # for training a GNN, we need the graph notes, i.e., the individual hits, as the first dimension,
        # so we need to transpose to get (n_hits, n_features)
        tensordata = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensordata = tensordata.to(dtype=torch.float32)

        # PyTorch Geometric needs the data in a specific format
        # we need to create a PyTorch Geometric Data object for each event
        this_graph_item = Data(x=tensordata)
        data_list.append(this_graph_item)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    labels = torch.cat(labels, dim=0) # convert the list of tensors to a single tensor
    packed_data = Batch.from_data_list(data_list) # convert the list of Data objects to a single Batch object
    return packed_data, labels