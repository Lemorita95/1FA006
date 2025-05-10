import torch
import time
import os

from plot import plot_train_loss

MAIN_PATH = os.path.dirname(os.path.realpath(__file__)) # change here
MODEL_PATH = os.path.join(MAIN_PATH, 'models')
os.makedirs(MODEL_PATH, exist_ok=True)


def train_validate_model(model, diffusion, lr, n_epochs, train_loader, val_loader, device):
    """
    Trains and validates a given model using a diffusion process.
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        diffusion (callable): A function or module that computes the loss for the diffusion process.
        lr (float): Learning rate for the optimizer.
        n_epochs (int): Number of epochs to train the model.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (CPU or GPU) to perform computations on.
    Returns:
        tuple: A tuple containing:
            - train_losses (list): List of average training losses for each epoch.
            - val_losses (list): List of average validation losses for each epoch.
            - epoch (int): The last completed epoch.
    Notes:
        - The function saves the model's state dictionary after each epoch to a file.
        - Training and validation progress is printed to the console, including loss and time per batch/epoch.
    """


    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    prev_model_path = None  # Track the previous model file
    for epoch in range(n_epochs):
        # implement training loop. You get the loss by calling the diffusion function
        # `loss = diffusion(training_images)`
        epoch_start_time = time.time()
        diffusion.train()
        train_loss = 0

        batch_no = 0
        for batch_no, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device)

            loss = diffusion(batch_x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() 

            print(f"\t<train> Epoch: {epoch}, Batch: {batch_no}/{len(train_loader)} ({100*batch_no/len(train_loader):.2f}%), Loss: {train_loss:.6f}, Time: {(time.time() - epoch_start_time):.2f}s", end='\r')  # Overwrite batch line

        train_loss /= len(train_loader) # calulate loss per epoch
        train_losses.append(train_loss)

        # Validation loop
        diffusion.eval()
        val_loss = 0
        with torch.no_grad():  # Disable gradient computation for validation
            for batch_no, (batch_x, _) in enumerate(val_loader):
                batch_x = batch_x.to(device)

                # Forward pass
                loss = diffusion(batch_x)
                val_loss += loss.item()

                print(f"\t<validation> Epoch: {epoch}, Batch: {batch_no}/{len(val_loader)} ({100*batch_no/len(val_loader):.2f}%), Loss: {val_loss:.6f}, Time: {(time.time() - epoch_start_time):.2f}s", end='\r')  # Overwrite batch line

        val_loss /= len(val_loader)  # Average validation loss
        val_losses.append(val_loss)

        # Print progress (print every epoch)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss}, Duration = {epoch_duration:.2f}s", end='\r')  # Overwrite epoch line
        print()  # Clear line after last batch

        # Save model checkpoint
        model_path = os.path.join(MODEL_PATH, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)

        # Delete the previous model checkpoint if it exists
        if prev_model_path and os.path.exists(prev_model_path):
            os.remove(prev_model_path)

        # Update the previous model path to the current one
        prev_model_path = model_path

    return train_losses, val_losses, epoch


def new_model(model, diffusion, lr, n_epochs, train_loader, val_loader, device):
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

    # Train the model
    train_start_time = time.time()
    train_losses, val_losses, epoch = train_validate_model(
        model = model,
        diffusion = diffusion,
        lr = lr,
        n_epochs = n_epochs,
        train_loader = train_loader,
        val_loader = val_loader,
        device = device
    )

    train_duration = time.time() - train_start_time
    print(f"\nTraining completed in {train_duration:.2f} seconds")
    print(f"Training time: {train_duration/epoch:.2f} seconds/epoch\n")

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


def create_load_model(model_file, model, **kwargs):
    """
    Create or load a machine learning model.
    This function either creates a new model and trains it or loads an existing model 
    from the specified file. If a model file does not exist, it requires additional 
    parameters to train a new model. If a model file exists, the user is prompted to 
    decide whether to train a new model or load the existing one.
    Args:
        model_file (str): Path to the model file.
        model (torch.nn.Module): The PyTorch model to be trained or loaded.
        **kwargs: Additional keyword arguments:
            - diffusion (object, optional): Diffusion process object for training.
            - lr (float, optional): Learning rate for the optimizer.
            - n_epochs (int, optional): Number of training epochs.
            - train_loader (DataLoader, optional): DataLoader for training data.
            - val_loader (DataLoader, optional): DataLoader for validation data.
            - device (torch.device, optional): Device to run the model on (e.g., 'cpu' or 'cuda').
    Raises:
        ValueError: If required arguments for creating or training a new model are missing.
    Behavior:
        - If the model file does not exist, a new model is created and trained.
        - If the model file exists, the user is prompted to either train a new model 
          or load the existing one.
        - Training losses and validation losses are plotted after training a new model.
    Note:
        Ensure all required arguments are provided when creating or training a new model.
    """

    diffusion = kwargs.pop('diffusion', None)
    lr = kwargs.pop('lr', None)
    n_epochs = kwargs.pop('n_epochs', None)
    train_loader = kwargs.pop('train_loader', None)
    val_loader = kwargs.pop('val_loader', None)
    device = kwargs.pop('device', None)
    
    missing = [x for x in (diffusion, lr, n_epochs, train_loader, val_loader, device) if x is None]

    # create new model or load existing model
    if not os.path.exists(model_file):
        print("No model found, creating a new model...")
        if missing:
            raise ValueError("Missing required arguments for creating a new model. See create_load_model()")
        train_losses, val_losses = \
            new_model(
                model = model,
                diffusion = diffusion,
                lr = lr,
                n_epochs = n_epochs,
                train_loader = train_loader,
                val_loader = val_loader,
                device = device,
            )
        # plot model losses
        plot_train_loss(
            train_losses,
            val_losses,
            "MNIST",
        )

    else:
        new_model_input = input("A model exists, do you want to train a new the model? (y/n): ").strip().lower()
        print()

        if new_model_input == 'y':
            print("Preparing to train a new model...")
            if missing:
                raise ValueError("Missing required arguments for training a new model. See create_load_model()")
            train_losses, val_losses = \
                new_model(
                    model = model,
                    diffusion = diffusion,
                    lr = lr,
                    n_epochs = n_epochs,
                    train_loader = train_loader,
                    val_loader = val_loader,
                    device = device,
                )
            # plot model losses
            plot_train_loss(
                train_losses,
                val_losses,
                "MNIST",
            )

        # if user does not want to create a new model, load the existing model
        else:
            print("Preparing to load the model...")
            load_model(model_file, model)