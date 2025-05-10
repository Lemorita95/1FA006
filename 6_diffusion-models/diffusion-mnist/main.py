from denoising_diffusion_pytorch import Unet, GaussianDiffusion

from dataset import get_dataloader
from plot import generate_plot_samples
from helpers import os, torch, MODEL_PATH, create_load_model

# use random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
LEARNING_RATE = 4e-4
BATCH_SIZE = 128  # Batch size
N_EPOCHS = 100
IMAGE_SIZE = 28
TIME_STEPS = 1000
SAMPLING_TIMESTEPS = 250

DIM = 32
DIM_MULTS = (1, 2, 5)

def main(num_samples):
    # Check for available devices and select if available
    if torch.cuda.is_available():
        device = torch.device("cuda")       #CUDA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")        #Apple GPU
    else:
        device = torch.device("cpu")        #if nothing is found use the CPU
    print(f"Using device: {device}")

    model = Unet(
        dim = DIM,
        dim_mults = DIM_MULTS,
        flash_attn = False,
        channels = 1
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = IMAGE_SIZE,
        timesteps = TIME_STEPS,           # number of steps
        sampling_timesteps = SAMPLING_TIMESTEPS    # number of sampling timesteps (using ddim for faster inference [see ddim paper])
    ).to(device)

    # get the dataloaders
    train_loader, val_loader = get_dataloader(BATCH_SIZE)

    model_file = os.path.join(MODEL_PATH, 'model_epoch_99.pth')

    # arguments to create a new model
    model_kwargs = {
        'diffusion': diffusion,
        'lr': LEARNING_RATE,
        'n_epochs': N_EPOCHS,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'device': device
    }
    # create or load the model
    create_load_model(model_file, model, **model_kwargs)

    # generate samples
    print(f"\n\tGenerating {num_samples} samples...")
    generate_plot_samples(diffusion, num_samples)

if __name__ == "__main__": 
    main(num_samples=100)