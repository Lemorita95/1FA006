# Assignment 7 - Transformer
implementation of a Transformer to reconstruct the position of the neutrino interaction from the measured Cherenkov light. 

## content
### [dataset.py](dataset.py)
### [helpers.py](helpers.py)
### [main.py](main.py)
### [model.py](model.py)
### [plot.py](plot.py)

## reproducibility
1. Save parquet files locally;
2. Change path of variables [line 9](main.py) it should points to the directory where the data from step 1 is placed;
3. Run [main.py](main.py);

## approach
0. hyperparameters:
    > batch_size = 64 <br>
    > learning_rate = 0.8e-4 <br>
    > num_epochs = 50 <br>
    > D = 128 <br>
    > n_heads = 2 <br>
    > dim_feedforward = 64 <br>
    > n_layers = 2 <br>
1. Load data with awkward and normalize features and labels with [get_normaized_data()](helpers.py);<br>
2. Data is normalized through z-score normalization;<br>
3. Train, validation and test dataset is created with PyTorch Dataloader and a custom collate function [collate_fn_transformer()](helpers.py);<br>
4. Training loss function is defined as MSE;<br>
5. Gives the user the possibility to load a model (if found at models/model.pth);<br>
6. Default [model](model.py) is defined with (n_layers) TransformerEncoderLayer layers with (n_head) attention heads + residual connection + MLP + residual connectiona MLP output layer each a final Linear layer is used to get the output dimension;<br>
8. Model [TransformerEncoder.forward()](model.py) pass: 1) input embedding with a Linear layer that outputs tensor at dimension (batch_size x N, D) where N is dependent on data, 2) padding and masking; <br> 
9. Train and validate model [train_validade_model()](helpers.py);<br>
10. Test model [test_model()](helpers.py);<br>
11. Model outputs the predicted labels [TransformerEncoder.forward()](model.py); <br>
12. Denormalize predictions and true values [denormalize()](helpers.py); <br>
13. [Plot](plot.py) scatter of averages and residuals;<br>

## results
1. the training loop for the [hyperparameters](#approach) took around 12 second per epoch; <br>
2. the training had convergence; <br>
3. similar performance of [GNN](../4_gnn/model.py) for this dataset; <br>

## challenges
1. the simplest transformer with 2 attention heads, 2 layers ("BASE MODEL") performed very similar to GNN as in results, in losses (Transformer: 0.0892 x GNN: 0.0883), normal distribution of residuals as in computation time; <br>
2. changing attention head 2 -> 8 from base model did not improved much (training loss: 0.0892 -> 0.0819); <br>
3. changing number of layers 2 -> 3 from base model did not improved much (training loss: 0.0892 -> 0.0823); <br>
4. changing number of layers 2 -> 6 from base model did not improved much (training loss: 0.0892 -> 0.0839) but slower down (training time per epoch: 12s -> 18s) with early stop at 50% of num_epochs; <br>
5. best result so far by changind dim_feedforward 32 -> 64 (train loss: 0.0892 -> 0.0806, training time per epoch: 11.8s -> 12s);
6. final model was with the result of 5.
