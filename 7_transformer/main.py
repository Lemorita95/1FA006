import numpy as np

from dataset import IceCubeDataset
from helpers import os, MAIN_PATH, create_load_model, test_model, denormalize
from model import torch, TransformerEncoder
from plot import plot_scatter, plot_residuals_scatter, plot_residuals_distribution

 # path to the main directory
DATA_PATH = os.path.join(os.path.dirname(MAIN_PATH), '4_gnn', 'data') # path to the data directory
MODEL_PATH = os.path.join(MAIN_PATH, 'models')

# hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.8e-4
N_EPOCHS = 50

HYPERPARAMS_TXT = f"transformer_bs{BATCH_SIZE}_lr{LEARNING_RATE}_ep{N_EPOCHS}"


def main():
    # Check for available devices and select if available
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

    # Initialize and load the dataset
    dataset = IceCubeDataset(DATA_PATH, BATCH_SIZE)

    d_model = 128
    nhead = 2
    dim_feedforward = 64
    num_layers = 2
    input_dim = 3
    output_dim = 2

    # Initialize the Transformer model
    model = TransformerEncoder(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        input_dim=input_dim, # (t, x, y)
        output_dim=output_dim # (xpos, ypos)
    ).to(device)

    model_params_txt = f"_D{d_model}_heads{nhead}_hd{dim_feedforward}_layers{num_layers}"
    experiment_name = HYPERPARAMS_TXT + model_params_txt
    model_name = f'model{model_params_txt}.pth'
    model_file = os.path.join(MODEL_PATH, model_name)
    print(f"\nExperiment name: {experiment_name}")
    print(f"Model name: {model_name}\n")


    criterion = torch.nn.MSELoss()

    # arguments to create a new model
    model_kwargs = {
        'experiment_name': experiment_name,
        'criterion': criterion,
        'device': device,
        'learning_rate': LEARNING_RATE,
        'train_loader': dataset.train_loader,
        'val_loader': dataset.val_loader,
        'num_epochs': N_EPOCHS,
        'patience': 5,
    }
    # create or load the model
    create_load_model(model_file, model, **model_kwargs)

    # test the model
    test_predictions, true_labels = test_model(
        model,
        dataset.test_loader,
        criterion,
        device,
    )

    # convert test_predictions and true_labels to numpy arrays so they can be sliced
    test_predictions = np.concatenate(test_predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # denormalize the predictions and true labels
    test_predictions = denormalize(test_predictions, dataset.labels_mean, dataset.labels_std)
    true_labels = denormalize(true_labels, dataset.labels_mean, dataset.labels_std)

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

if __name__ == "__main__":
    main()