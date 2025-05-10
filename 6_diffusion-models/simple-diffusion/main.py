import matplotlib.pyplot as plt
import time
import torch.optim as optim

from model import DiffusionModel
from plot import np, torch, plot_train_loss, plot_results

# # This is a simple example of a diffusion model in 1D.

# use random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# generate a dataset of 1D data from a mixture of two Gaussians
# this is a simple example, but you can use any distribution
data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor([1, 2])),
    torch.distributions.Normal(torch.tensor([-4., 4.]), torch.tensor([1., 1]))
)
experiment_name = "two_gaussians"  # name of the experiment

dataset = data_distribution.sample(torch.Size([10000]))  # create training data set
dataset_validation = data_distribution.sample(torch.Size([1000])) # create validation data set

# we will keep these parameters fixed throughout
# these parameters should give you an acceptable result
# but feel free to play with them
TIME_STEPS = 250
BETA = torch.tensor(0.02)
N_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.8e-4

# define the neural network that predicts the amount of noise that was
# added to the data
# the network should have two inputs (the current data and the time step)
# and one output (the predicted noise)

g = DiffusionModel()  # create the model
optimizer = optim.Adam(g.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()  # mean squared error loss

train_losses, val_losses = [], []

# epochs = tqdm(range(N_EPOCHS))  # this makes a nice progress bar
train_start_time = time.time()
for e in range(N_EPOCHS): # loop over epochs
    g.train()
    train_loss = 0
    # loop through batches of the dataset, reshuffling it each epoch
    indices = torch.randperm(dataset.shape[0])
    shuffled_dataset = dataset[indices]
    n_batches_train = 0 # to track number of batches for loss calculation

    epoch_start_time = time.time()
    for i in range(0, shuffled_dataset.shape[0] - BATCH_SIZE, BATCH_SIZE):
        # sample a batch of data
        x0 = shuffled_dataset[i:i + BATCH_SIZE]

        # here, implement algorithm 1 of the DDPM paper (https://arxiv.org/abs/2006.11239)
        
        # sample a random time step t
        t = torch.randint(low=1, high=TIME_STEPS, size=(BATCH_SIZE,))

        # sample a random noise vector
        random_noise = torch.randn_like(x0) # sample random noise from a normal distribution
        
        # set the gradients to zero
        optimizer.zero_grad() 

        # get noisy samples
        alpha_t_bar = (1 - BETA) ** t
        xt = torch.sqrt(alpha_t_bar) * x0 + torch.sqrt(1 - alpha_t_bar) * random_noise # add noise to the data

        # add dimension for concatenation
        xt = xt.view(xt.size(0), 1)
        t = t.view(t.size(0), 1)
        random_noise = random_noise.view(random_noise.size(0), 1)

        # Forward pass
        predictions = g(torch.cat((xt, t), dim=1)) # make a prediction with the current model, uses the forward() method
        loss = criterion(predictions, random_noise) # calculate the loss based on the prediction

        # Backward pass and optimization
        loss.backward() # calculated the gradiets for the given loss
        optimizer.step() # updates the weights and biases for the given gradients
        train_loss += loss.item() # calulate loss per batch
        n_batches_train += 1

    train_loss /= n_batches_train # calulate loss per epoch
    train_losses.append(train_loss)

    # # compute the loss on the validation set
    g.eval() # The model should be in eval mode to not use batch normalization and dropout
    val_loss = 0

    # loop through batches of the dataset, reshuffling it each epoch
    indices_validation = torch.randperm(dataset_validation.shape[0])
    shuffled_dataset_validation = dataset_validation[indices_validation]
    n_batches_val = 0 # to track number of batches for loss calculation

    with torch.no_grad(): # make sure the gradients are not changed in this step
        xx = list(range(0, shuffled_dataset_validation.shape[0] - BATCH_SIZE, BATCH_SIZE))
        for i in range(0, shuffled_dataset_validation.shape[0] - BATCH_SIZE, BATCH_SIZE):
            # sample a batch of data
            x0 = shuffled_dataset_validation[i:i + BATCH_SIZE]

            # sample a random time step t
            t = torch.randint(low=1, high=TIME_STEPS, size=(BATCH_SIZE,))

            # sample a random noise vector
            random_noise = torch.randn_like(x0) # sample random noise from a normal distribution

            # get noisy samples
            alpha_t_bar = (1 - BETA) ** t
            xt = torch.sqrt(alpha_t_bar) * x0 + torch.sqrt(1 - alpha_t_bar) * random_noise # add noise to the data

            # add dimension for concatenation
            xt = xt.view(xt.size(0), 1)
            t = t.view(t.size(0), 1)
            random_noise = random_noise.view(random_noise.size(0), 1)

            # forward pass
            predictions = g(torch.cat((xt, t), dim=1)) # make a prediction with the current model
            loss = criterion(predictions, random_noise) # calculate the loss based on the prediction
            val_loss += loss.item() # calulate loss per batch

            n_batches_val += 1
        
    val_loss /= n_batches_val # calulate loss per epoch
    val_losses.append(val_loss)

    # Print progress (print every epoch)
    epoch_duration = time.time() - epoch_start_time
    if e % 1 == 0:
        print(f"Epoch {e}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Duration = {epoch_duration:.2f}s")
print(f"Total training time: {time.time() - train_start_time:.2f}s")


def sample_reverse(g, count):
    """
    Sample from the model by applying the reverse diffusion process

    Here, implement algorithm 2 of the DDPM paper (https://arxiv.org/abs/2006.11239)

    Parameters
    ----------
    g : torch.nn.Module
        The neural network that predicts the noise added to the data
    count : int
        The number of samples to generate in parallel

    Returns
    -------
    x : torch.Tensor
        The final sample from the model
    """
    g.eval()
    sampling_start_time = time.time()

    xt = torch.normal(mean=0, std=1, size=(count, 1))  # sample from the normal distribution

    for t in range(TIME_STEPS, 0, -1):
        z = torch.normal(mean=0, std=1, size=(count, 1)) if t > 1 else 0  # sample from the normal distribution
        alpha_t = (1 - BETA)
        alpha_t_bar = (1 - BETA) ** t
        t_tensor = torch.tensor([t] * count).view(count, 1)

        xt_1 = (1 / torch.sqrt(alpha_t)) * (xt - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_bar)) * g(torch.cat((xt.view(xt.size(0), 1), t_tensor), dim=1))) + torch.sqrt(BETA) * z

        # update xt
        xt = xt_1
    
    print(f"Total sampling time: {time.time() - sampling_start_time:.2f}s")
    return xt

plot_train_loss(train_losses, val_losses, experiment_name)

samples = sample_reverse(g, 100000)
samples = samples.detach().numpy()

plot_results(dataset, samples, experiment_name)