# Assignment 3 - Normalizing Flows
Implementation of a neural network that predicts the temperature, gravity, and metallicity of stars and the uncertainties of predictions using Normalizing Flows using the negative log-likelihood loss function.

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
    > batch_size = 64 <br>
    > learning_rate = 0.8e-4 <br>
    > num_epochs = 10 <br>
1. Load data and stored in a [CustomDataset](dataset.py) class;<br>
2. Data is normalized within CustomDataset class \__init__ through z-score normalization;<br>
3. Train, validation and test dataset is created with PyTorch Dataloader;<br>
4. Training loss is defined as negative mean of log probability [nf_loss()](helpers.py);<br>
5. Gives the user the possibility to load a model (if found at models/model.pth);<br>
6. Default [model](model.py) is defined with 6 convolution layers (ReLU activation), 2 hidden layers (ReLU activation) and 1 output layers;<br>
7. Output layer with variable size due to Normalizing flow; <br>
8. Train and validate model [train_validade_model()](helpers.py);<br>
9. Test model [test_model()](helpers.py);<br>
10. Model outputs the mean and standard deviation of samples [CombinedModel.forward()](model.py); <br>
11. Denormalize mean and standard deviation [denormalize(), denormalize_std()](helpers.py), for std the mean for normalizing is not added; <br>
12. [Plot](plot.py) pull histogram, scatter of averages and residuals;<br>

## results
1. the training-validation loop for the [hyperparameters](#approach) took around 5 minutes to complete the diagonal gaussian, 5 minutes for full gaussian and 1:50h for full flow; <br>
2. in terms of train and validation loss, the model only outperforms the TinyCNN for the full flow; <br>
3. model does not improve validation loss (overfits) for full gaussian after 10 epochs. <br>
4. model behavior is unstable for different flows. So the use of the same model and hyperparameter for different flows are limited. <br>

## challenges
1. tried two different models, CNNModel (from previous tasks) and TinyCNN - CNN runs faster but overfits easilier for some flows. <br>
2. for this task, the models seem to be more sensitive to learning rate, for CNNModel i had to lower it for convergence for diagonal gaussian and full gaussian, for full flow i was able to increase the learning rate and batch size. <br>
3. two different models for different flows presented a correlation between true values and residuals. The  effect was not completedly understood whether its caused by the data processing, hyperparameter tuning or model architecture.
4. **diagonal gaussian flow**: CNNModel overfits after 10 epochs. changed the architecture of model (maxpool -> avgpool), included global_pool before fully connected layer and the model performed better, did not overfitted but the validation loss did not got much better after 10th epoch. TinyCNN runs slowly but performs good and consistently. given the difference at validation loss (CNNModel: 3.53 x TinyCNN: -0.04) and time to train 10 epochs (CNNModel: 11 min x TinyCNN: 19 min), TinyCNN is preferable. <br>
5. **full gaussian flow**: after overfit adjustments at CNNModel, the same behavior as in diagonal gaussian flow was observed for both models with TinyCNN outperforming (in terms of validation/test loss) CNNModel. <br>
6. **full flow**: computational expensive, even on google colab. CNNModel took 4 hours (20 epochs), the training and validation loss was continuously decreasing and the slope was constant, not decreasing, seems that if more epochs was performed, the model would be better. the mean predictions however, was not very meaningfull (predicting constant values). When increase the batch size (64) and learning rate (0.8e-4) for CNN model, the performance got better (from CNNModel with batch_size=32 and lr=0.8e-5 and from TinyCNN) but the computation time did not changed much (for 20 epochs, 1st CNNModel: 3.8h, TinyCNN: 3.78h, CNNModel2: 3.5h). <br>
7. The predicted means for full flow, however, was kinda far from the perfect predictions (x=y). And the residuals for log_g presented a high correlation to the true values.<br>
8. the final model of choice was CNNModel with batch_size=64 and lr=0.8e-4. <br>
9. for full flow it was harder to tune hyperparameters due to the amount of time it took for training, so not much further investigation was made. <br>
10. The same CNNModel for full flow does not work optimally for diagonal and full gaussian, starts to overfit after ~17 epochs and predicted means correlation to true values are low. <br>
11. Tried softplus activation for hidden layers, but the model did not converged. <br>
12. Altough percentile normalization could increase the performance of the model when handling extreme values, this test was not made for the model of this assignment. <br>