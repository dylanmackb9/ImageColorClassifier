
import os
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def dNN(x_train, y_train, activ, lr, epo):  
    # sequential fully connected neural network 
    # one hidden layers, each with a width of 2 nodes

    # Iterable over activation function, learning rate, and training epochs, softmax output

    # x_train: training data with input size
    # y_train: labels for training data
    # activ: activation function for hidden layers
    # lr: learning rate for optimizer 
    # epo: number of training epochs
	seqNN = keras.Sequential(
        [
            keras.Input(shape=(1,)), # input shape of 1, for a single feature
            layers.Dense(2, activation=activ, name='layer1'),  # 2 hidden layers
            layers.Dense(2, activation='softmax'),  # softmax on output
        ]
    )
    
	seqNN.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr),  # Using Adam (SGD) as iterative optimizer
        loss = keras.losses.SparseCategoricalCrossentropy(),  # Using SparseCategoricalCrossEntropy as loss
        metrics = ['accuracy'],
    )
        
	seqNN.fit(x_train, y_train, epochs=epo, verbose=0)  # fitting 


	return seqNN



