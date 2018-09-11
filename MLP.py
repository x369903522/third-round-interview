from keras.models import Sequential
from keras.layers import Dense, Dropout
from input import *
from keras import metrics
import numpy as np

def mlp(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(100, input_dim=20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, validation_split=0.2, epochs=100, verbose=0, batch_size=128)
    score = model.evaluate(X_test, y_test)
    print score



all_data, all_labels, vali_data, vali_labels = read_all_data_from_csv()

model = mlp(np.reshape(all_data,(all_data.shape[0],-1)),np.reshape(vali_data,(vali_data.shape[0],-1)),all_labels,vali_labels)
