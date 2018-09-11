# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/ResNeXt-in-tensorflow
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = 'data'
# full_data_dir = 'cifar10_data/cifar-10-batches-py/data_batch_'
# vali_dir = 'cifar10_data/cifar-10-batches-py/test_batch'
# DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


IMG_WIDTH = 20
IMG_HEIGHT = 1
IMG_DEPTH = 1
# NUM_CLASS = 10

TRAIN_RANDOM_LABEL = True # Want to use random label for train data?
VALI_RANDOM_LABEL = True # Want to use random label for validation?

# NUM_TRAIN_BATCH = 5 # How many batches of files you want to read in, from 0 to 5)
EPOCH_SIZE = 2263


def read_all_data_from_csv():
    data = pd.read_csv("data/Dataset.csv",index_col=0)
    input = data.values * 100
    X, y = input[:, 0:20], input[:,20]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 4)

    train_num_data = len(X_train)
    X_train_pic = X_train.reshape((train_num_data, 1, 20, IMG_DEPTH))

    valid_num_data = len(X_test)
    X_test_pic = X_test.reshape((valid_num_data, 1, 20, IMG_DEPTH))

    # pad_width = ((0, 0), (0, 0), (2, 2), (0, 0))
    # X_train_pic = np.pad(X_train_pic, pad_width=pad_width, mode='constant', constant_values=0)
    # X_test_pic = np.pad(X_test_pic, pad_width=pad_width, mode='constant', constant_values=0)

    return X_train_pic, y_train, X_test_pic, y_test

