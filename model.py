
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



def cNN(x_train, y_train, activ, lr, epo, kernel_size, stride):
    # convolutional neural network 
    # 3 convolution->pool layers, 2 dense network layers 
    # Iterable over activation function, learning rates, training epochs, and convolution kernel size

    # x_train: training data with input size
    # y_train: labels for training data
    # activ: activation function for hidden layers
    # lr: learning rate for optimizer 
    # kernel_size: size of convolution filter
    # stride: stride movement of filter
    convNN = keras.Sequential(
        [
            keras.Input(shape=(30, 30, 4)),  # input of 30x30x4 images
            layers.Conv2D(2, 3, padding='valid', activation=activ),  # now we have 2x28x28x4
            layers.MaxPool2D(pool_size=(2,2)),  # no we have 2x14x14x4
            layers.Conv2D(4, 3, padding='valid', activation=activ),  # now we have 8x12x12x4
            layers.Flatten(),
            layers.Dense(64, activation=activ, name='finaldense'),
            layers.Dense(2),

        ]
    )

    print(convNN.summary())

    convNN.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # logits giving softmax on output layer
        optimizer = keras.optimizers.Adam(learning_rate=lr),
        metrics = ['accuracy'],
    )

    convNN.fit(x_train, y_train, epochs=epo, verbose=0)

    return convNN






