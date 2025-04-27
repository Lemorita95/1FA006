# Assignment 5 - Generative Models
Design and train a Generative Adversarial Network (GAN) to generate handwritten digits using the MNIST dataset

## content
### [helpers.py](helpers.py)
### [main.py](main.py)
### [model.py](model.py)

## reproducibility
1. Run [main.py](main.py);

## approach
0. hyperparameters:
    > batch_size = 64 <br>
    > learning_rate = 2e-4 <br>
    > num_epochs = 50 <br>
    > latent_dimension = 128 <br>
    > optimizer $ \beta_1 $ = 0.5
1. Define a transform that converets image to tensor and normalize it with mean=0.5 and std=0.5 (convertes image range [0, 1] to [-1, 1], suitable for tanh activation function of generator); <br>
2. Load data with torchvision dataset and apply transform;<br>
3. Load dataset into a DataLoader object; <br>
4. To develop the [models](model.py), their architecture were based on the article https://arxiv.org/pdf/1511.06434; <br>
5. Instantiate [Discriminator()](model.py) with number of channels of image (1 for MNIST, greyscale); <br>
6. Instantiate [Generator()](model.py) with latent_dimension and number of channels of image; <br>
7. Define optimizer with same learning rate and $ \beta_1 $ for Discriminator and Generator; <br>
8. Define loss function (BCE for binary classification); <br>
9. [Train](helpers.py) model and log training progress in tensorboard; <br>

## results
1. the training loop for the [hyperparameters](#approach) took around 2.5 hours; <br>
2. the training had convergence; <br>
3. the generator is able to generate images of digits, altough some digits might look clearer for an human interpretation (numbers 0, 3, 6, 7, 9); <br>

## challenges
1. Computational expensive to train the network (almost 3h); <br>
2. Runned the model with default values of Adam optimizer $ \beta_1 $ and the train did not converge, the discriminator dominated at the adversarial approach (very low loss but high loss for generator); <br>
3. Then, following the guidelines of https://arxiv.org/pdf/1511.06434, page 3, "Details of Adversarial Training", the training process did converged when changing the $ \beta_1 $ parameter to 0.5 at Discriminator and Generator optimizers; <br>
4. Further improvements that could be made (not tested): <br>
- decrease the learning rate of Discriminator (since its dominating the training). <br>
- change the models architecture (more complex for Generator and less complex for Discriminator). <br>
- implement Wasserstein distance as the loss function and remove the final sigmoid activation from the model. <br>