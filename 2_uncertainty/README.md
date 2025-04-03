# Assignment 2 - Uncertainty Prediction
Implementation of a neural network that predicts the temperature, gravity, and metallicity of stars and the (Gaussian) uncertainties of predictions using the negative log-likelihood loss function.

## content
### [dataset.py](dataset.py)
### [helpers.py](helpers.py)
### [main.py](main.py)
### [model.py](model.py)
### [plot.py](plot.py)

## reproducibility
1. Download data at https://huggingface.co/datasets/simbaswe/galah4/tree/main, labels.npy and spectra.npy;
2. Change path of variables [location and DATA_DIR](main.py) it should points to the directory where the data from 1. it placed;
3. Run [main.py](main.py);

## approach
0. hyperparameters:
    > batch_size = 32 <br>
    > learning_rate = 2e-4 <br>
    > num_epochs = 10 <br>
1. Load data and stored in a [CustomDataset](dataset.py) class;<br>
2. Data is normalized within CustomDataset class \__init__ through z-score normalization;<br>
3. Train, validation and test dataset is created with PyTorch Dataloader;<br>
4. Training loss is defined as Negative Loss-likelihood at [nll_loss()](helpers.py);<br>
5. Gives the user the possibility to load a model (if found at models/model.pth);<br>
6. Default [model](model.py) is defined with 5 convolution layers (ReLU activation), 2 hidden layers (Softmax activation) and 1 output layers;<br>
7. Output layer with 6 neurons (2 for each label), 3 for mean and 3 for log(standard deviation); <br>
8. Train and validate model [train_validade_model()](helpers.py);<br>
9. Test model [test_model()](helpers.py);<br>
10. [Plot](plot.py) pull histogram and scatter of averages;<br>

## results
1. the training-validation loop for the [hyperparameters](#approach) took around 2.2 minutes to complete; <br>
2. training loss becomes negative in the 2nd epoch because of log(standard deviation) and is negatively larger then 0.5*log(2*pi); <br>
3. training loss has a monotonic decrease throughout the epochs. <br>
4. validation losses decreases throughout epochs.<br>

## challenges
1. training model are giving negative loss for some activation function; <br>
2. when scatter plotting predicted mean vs true values with some activation functions (tanh, leakyrelu) some points with zero correlation can be seen, they also have high standard deviation (~0.98); <br>
3. when using tanh as activation function for the last layer, pull histogram seems better but model predicts always predicts the same mean value - e.g. log(sigma) being more relevant on the loss function; <br>
4. using percentile normalization also does not improved model prediction of mean and standard deviation, labels have higher bias; <br>
5. softplus is the better so far; <br>
6. softmax prevent negative losses but predicts discrete mean values; <br>
7. lower learning rate prevent model for making a very distance first batch guess; <br>
8. adding l2 regularization prevent model for not learning the mean (guessing zero mean); <br>
9. adding more channels on data increase the performance but the time spent on training increased a lot; <br>
10. adding more complexity to model increase its quality; <br>
11. final configuration considered a good enough quality and time of execution of training; <br>
12. the final model configuration overfitts around after 15 epochs; <br>