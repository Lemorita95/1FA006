# Assignment 6.2 - Diffusion MNIST
Design and training of a Denoising Diffusion Probabilistic Model (DDPM) to generate handwritten digits using the MNIST dataset.

## content
### [dataset.py](dataset.py)
### [helpers.py](helpers.py)
### [main.py](main.py)
### [plot.py](plot.py)

## reproducibility
1. Run [main.py](main.py);

## approach
0. hyperparameters:
    > batch_size = 128 <br>
    > learning_rate = 4e-4 <br>
    > num_epochs = 100 <br>
    > diffusion time_steps = 1000 <br>
    > diffusion sampling_timesteps = 250 <br>
1. Uses [Unet](main.py) and [GaussianDifussion](main.py) from denoising_diffusion_pytorch; <br>
2. [Load](dataset.py) MNIST dataset as a pytorch DataLoader; <br>
3. Train model with [train_validate_model()](helpers.py); <br>
4. [Generate](plot.py) image samples; <br>


## results
1. the training loop for the [hyperparameters](#approach) took around 7.5 minutes per epoch; <br>
2. the training had convergence; <br>
3. most of digits were easy to identify through human perception; <br>
4. generating new data took around 0.5s/sample; <br>

## challenges
1. easier to implement training (compared to the simple diffusion) thanks to denoising_diffusion_pytorch; <br>
2. laborous training, however the results were good, slightly better then compared to the [GAN](../../5_generative-models/main.py) implemented; <br>
3. did not implement tensorboard, it would be nice to see the image generation evolving; <br>
