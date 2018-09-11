# ResNeXt 

For the sake of visualization, I amplify the features and labels 100 times. 
I treat each example as an image with the same size of 1 * 20 without padding and use mean square loss.
Actually, I have tried padding but it does not help.
252 examples are reversed for validation.

## Quick guide

Just need to run the train.py, the program will print the square loss of training and validation. 

## Result discussion

Based on my experience, the ResNeXt has the mean square loss of 5.05 on the validation set. 
Meanwhile, the MLP roughly has the mean square loss of 6.13 on the same validation set.

### Files included
1. hyper-parameters.py defines the hyper-parameters related to train, ResNeXt structure, etc.

2. input.py includes the data I/O

3. resNeXt.py is the main body of ResNeXt network

4. train.py is responsible for the training and validation

5. MLP.py is the baseline model.