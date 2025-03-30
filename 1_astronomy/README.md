# Assignment 1 - Astronomy CNN with PyTorch
In this assignment we are working with the GALAH Star Spectra Dataset that contains 8,914 high-resolution stellar spectra, each with 16,384 flux measurements spanning the wavelength range from 4718 Å to 7890 Å. <br> 
The task is to implement a Convolutional Neural Network (CNN) to predict the Surface Temperature, Surface Gravity and Metallicity of a star given its stellar spectra.

## content
### [dataset.py](dataset.py)
### [main.py](main.py)
### [model.py](model.py)

## approach
0. hyperparameters:
    > batch_size = 32 <br>
    > learning_rate = 0.001 <br>
    > num_epochs = 10 <br>
1. the device used to alocate pytorch tensors was MPS. <br>
2. data was stored locally (due to its size) in a folder called data/ at a .npy extension. the data can be found at https://huggingface.co/datasets/simbaswe/galah4/tree/main. <br>
3. a CustomDataset class ([CustomDataset()](dataset.py)) is created to handle the dataset, it inherits the Dataset class from torch.utils.data. <br>
3.1 features and labels normalization was handle during the class instantiation. features (wavelength) were logarithm normalized and labels (surface temperature, surface gravity and metallicity) were normalized using standard score. <br>
3.2 as the class take data path and load files as numpy arrays, this class also convert the arrays to tensors only when an element is called using the \__getitem__ method. the tensors is sent to device during each batch at train, validation and test phases. <br>
4. with the help of torch.utils.data Dataloader() train (70%), validation (15%) and test (15%) datasets are handled. <br>
5. the model [(model.py)](model.py) consists of 3 convolution-maxPooling layers with ReLu activation function, one fully connected hidden layer with ReLu activation and the output layers with 3 neurons. data is flattened before passing to the fully connected layer. <br>
5.1 the loss function used (criterion) is the Mean Squared Error (MSE). <br>
5.2 the chosen optimization method is Adam. <br>
6. training, validation and test is made and the values of interest recorded and a python variables and also logged using tensorboard. (tensorboard --logdir=runs) <br>

## results
1. the training-validation loop for the [hyperparameters](#approach) took around 2.5 minutes to complete. <br>
2. training loss has a monotonic decrease throughout the epochs. <br>
3. validation losses decreases throughout epochs.

## challenges
the experiments below are made changing only the informed [hyperparameter](#approach) keeping the others constant. <br>
when talking about performance, it is refered to: train and validation loss.
1. Learning rate: using a higher value (0.01) leads to overshoot and worst model performance. <br>
2. Batch size: lower batch sizes leads to lower train and validation losses at the initial epochs but for values as 16 and 64 it converges to similar performance at the 10th epoch. <br>
3. Epochs: from the 10th to the 15th epoch, there is no significant increase in the model performance and the validation losses presents a movement to increase (overfitting).
4. Model loading: for further assignments, a function to load a model (if any) will be implement to save redundant computations, expecially when acquiring images and plots outside tensorbord.
