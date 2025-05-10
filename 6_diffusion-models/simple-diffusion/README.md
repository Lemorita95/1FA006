# Assignment 6.1 - Simple Diffusion
Implementation of a diffusion network from scratch for the simplest possible case of a one-pixel image, i.e., just a single value. This value follows the following distribution given by a mixture of Gaussians.

## content
### [main.py](main.py)
### [model.py](model.py)
### [plot.py](plot.py)

## reproducibility
0. **optional** change [data_distribution](main.py)
1. Run [main.py](main.py);

## approach
0. hyperparameters:
    > batch_size = 64 <br>
    > learning_rate = 0.8e-4 <br>
    > num_epochs = 1000 <br>
    > diffusion $ \beta_1 $ = 0.02 <br>
    > diffusion time_steps = 250 <br>
1. Define the target [data_distribution](main.py) that we want to predict 
2. Sample train and validation dataset from it; <br>
3. Create model [DiffusionModel()](model.py) - used a simple MLP; <br>
4. The model takes 2 **inputs**: current sample value and the time step and **outputs**: amount of noise added in timestep t; <br>
5. Loss function is defined as MSE; <br>
6. Training loop: <br>
    1) sample a random timestep (`t`); <br>
    2) sample a random from from a standard normal distribution (`e`); <br>
    3) estimate noise (`e(xt, t)`) for the 'clean' data (`x0`) at timestep (`t`); <br>
    4) compute loss between sampled random noise (`e`) and estimated loss (`e(xt, t)`); <br>
7. Generate new data with [sample_reverse()](main.py); <br>
8. Compare with [data_distribution](main.py); <br>

## results
1. the training loop for the [hyperparameters](#approach) took around 45 seconds; <br>
2. the training had convergence; <br>
3. fast to produce new data for this simple exemple, 100k samples in 3.7 seconds; <br>
4. testes with two diffent distributions and worked well; <br>

## challenges
1. high initial loss but reduces fast, not much improvement after 500 epochs; <br>
