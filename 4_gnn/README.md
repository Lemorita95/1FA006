# Assignment 4 - GNN
Dynamic Edge Convolution Graph Neural Network to reconstruct the position of the neutrino interaction from the measured Cherenkov light.

## content
### [helpers.py](helpers.py)
### [main.py](main.py)
### [model.py](model.py)
### [plot.py](plot.py)

## reproducibility
1. Save parquet files locally;
2. Change path of variables [line 10](main.py) it should points to the directory where the data from step 1 is placed;
3. Run [main.py](main.py);

## approach
0. hyperparameters:
    > batch_size = 64 <br>
    > learning_rate = 0.8e-4 <br>
    > num_epochs = 50 <br>
    > k_neighbors = 10 <br>
    > n_layers = 2 <br>
1. Load data with awkward and normalize features and labels with [get_normaized_data()](helpers.py);<br>
2. Data is normalized through z-score normalization;<br>
3. Train, validation and test dataset is created with PyTorch Dataloader and a custom collate function [collate_fn_gnn()](helpers.py);<br>
4. Training loss function is defined as MSE;<br>
5. Gives the user the possibility to load a model (if found at models/model.pth);<br>
6. Default [model](model.py) is defined with 2 dynamic edge convolution layers with a MLP kernel and a MLP output layer;<br>
7. Edge convolution MLP kernel have a linear input layer, a fully connected hidden layer with 128 neurons and a linear output layer. Between each edgeconv layer the number of channels are kept constant at 128 neurons; <br>
8. Train and validate model [train_validade_model()](helpers.py);<br>
9. Test model [test_model()](helpers.py);<br>
10. Model outputs the predicted labels [GNNEncoder.forward()](model.py); <br>
11. Denormalize predictions and true values [denormalize()](helpers.py); <br>
12. [Plot](plot.py) scatter of averages and residuals;<br>

## results
1. the training-validation loop for the [hyperparameters](#approach) took around 14 seconds per epoch; <br>
2. the best model was from the current hyperparameters and MLP architecture, both in terms of validation loss and computation time; <br>
3. relatively simple model with MLP and EdgeConv layers flexibility for further accessments; <br>
4. no signs of overfitting at train/validation losses for 50 epochs;
5. optimal number of neighbors [(k_neighbors)](main.py) and DynamicEdgeConv layers [(n_layers)](main.py) found was 10 and 2 respectively. 

## challenges
1. Awkard arrays are more complex then numpy arrays for debugging but the operations (normalization) ware quite straightforward; <br>
2. Some hard-coded keys from data normalization -> not optimal; <br>
3. changing batch_size (32->64) did not impact much on performance (losses and computation time) - maintaning other parameters constant;<br>
4. 10 neighbors had lower validation loss and was more computation efficient, comparing to 5 and 10 neighbors - maintaning other parameters constant;<br>
5. changing the number of DynamicEdgeConv layers did impact mostly on computation time, not at losses - maintaning other parameters constant;<br>
6. DynamicEdgeConv fully connected layer number of channels had significant impact on both validation loss and computation time, more neurons yield less losses but more computation expensive (non-linear increse in time with linear-ish decrease in loss); <br>
