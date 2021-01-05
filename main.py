


import os
import numpy as np
import statistics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from sklearn.utils import shuffle 
from PIL import Image
from warnings import simplefilter

import model as nn # files

simplefilter(action='ignore', category=FutureWarning)  # ignoring future warnings from sklearn
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)  # ignoring warnings from tf2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


featurePath = "/Users/Dylan/Documents/python/project_drafts/imagePractice/dataset_features.npy"  # grabbing path of data
labelPath = "/Users/Dylan/Documents/python/project_drafts/imagePractice/dataset_labels.npy"  # grabbing path of data

dataset_train_x = np.load(featurePath)
dataset_train_y = np.load(labelPath)





# FEATURE EXTRACTION

dataset_features = dataset_train_x / 255  # dividing all pixel values by 255
dataset_labels = dataset_train_y


sumfeature = np.zeros((1000,1))  # initializing sum feature to show the sum of each example

for i in range(1000):
	sumfeature[i] = np.sum(dataset_features[i])




# FEATURE PROCESSING

features_d = sumfeature  # feature matrix for extracted on dNN
features_c = dataset_features  # feature matrix for pixels on cNN
labels = dataset_labels

#testing
#print(features[0])
#print(np.where(features==459000)[0].shape)
#print(np.hstack((features, labels.reshape((1000,1))))[0:50])

# training and test sets for sequential network
x_train_d = features_d[0:700]  
y_train_d = labels[0:700]


x_test_d = features_d[700:]
y_test_d = labels[700:]


# training and test sets for convolutional network 
x_train_c = features_c[0:700]
y_train_c = labels[0:700]


x_test_c = features_c[700:]
y_test_c = labels[700:]



# k-fold cross val

def kcross_dnn(k, lr, epo, activ):
    # Implementing k-fold cross validation for sequential neural network 

    # k: number of splits for cross val
    # lr: learning rate 
    # epo: number of training epochs
    # activ: activation function used

    m = int(x_train_d.shape[0])  # number of training examples
    kaccuracy_list = []
    
    for i in range(k):  # cross val training 

        i = tf.convert_to_tensor(i, dtype=tf.int64)
        x_val = x_train_d[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        y_val = y_train_d[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        
        x = np.vstack((x_train_d[0:i*m//k],x_train_d[(i*m//k+m//k):]))  # setting training to be everything but val
        y = np.concatenate((y_train_d[0:i*m//k],y_train_d[(i*m//k+m//k):]))
        
        curNN = nn.dNN(x, y, activ, lr, epo)  # training nn with created training set above  

        loss, accuracy = curNN.evaluate(x_val, y_val, verbose=0)  # calculating loss and accuracy on given k-fold 
        
        kaccuracy_list.append(accuracy)  # appending one of the k accuracies to a list
        
    average_kacc = statistics.mean(kaccuracy_list)  # averaging all k accuracies 
    
    return average_kacc

def kcross_cnn(k, lr, epo, activ):
    # Implementing k-fold cross validation for convolutional neural network 

    # k: number of splits for cross val
    # lr: learning rate 
    # epo: number of training epochs
    # activ: activation function used

    m = int(x_train_c.shape[0])  # number of training examples
    kaccuracy_list = []
    
    for i in range(k):  # cross val training 

        i = tf.convert_to_tensor(i, dtype=tf.int64)
        x_val = x_train_c[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        y_val = y_train_c[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        
        x = np.vstack((x_train_c[0:i*m//k],x_train_c[(i*m//k+m//k):]))  # setting training to be everything but val
        y = np.concatenate((y_train_c[0:i*m//k],y_train_c[(i*m//k+m//k):]))
        
        curNN = nn.cNN(x, y, activ, lr, epo)  # training nn with created training set above  

        loss, accuracy = curNN.evaluate(x_val, y_val, verbose=0)  # calculating loss and accuracy on given k-fold 
        
        kaccuracy_list.append(accuracy)  # appending one of the k accuracies to a list
        
    average_kacc = statistics.mean(kaccuracy_list)  # averaging all k accuracies 
    
    return average_kacc



# GRID SEARCH

#Hyperparameters for NN
lr_range_d = [.001,.01, .1, 1, 10]
epochs_d = [3, 5, 10, 15, 20, 30]
activation_d = ['linear','relu']

lr_range_c = [.00001, .0001, .001, .01, .1, 1]
epochs_c = [3, 5, 10, 15, 20, 30]
activation_c = ['relu']
# grid search not currently supporting kernel or stride for CNN

k = 10 # based on 700 size training set


gridsearchList = []
gridsearchDict = {}

def gridSearch():
    # Implementing a Grid Search for nn model tuning on give hyperparameters


    for activ in activation:
        for lr in lr_range:
            for ep in epochs:
                accuracy = kcross_cnn(k, lr, ep, activ)  # taking cross val accuracy from cnn or dnn
                gridsearchList.append(accuracy)
                gridsearchDict[(activ,str(lr), str(ep))] = accuracy 
                #print("For "+activ+" activation and "+str(lr)+" learning rate and "+str(ep)+" epochs: "+str(accuracy))

    return max(gridsearchList), gridsearchDict



maxvalue, historyDict = gridSearch()

print(maxvalue)
#print(historyDict.values())
print(list(historyDict.keys())[list(historyDict.values()).index(maxvalue)])  # ideal hyper parameters 


# Optimal hyperparaetmers for
#   dnn: 
#        activ: 'linear'
#        lr: .1
#        epochs: 8

#   cnn:
#        activ: 'relu'
#        lr: .0001
#        epochs: 5
    

idealmod_d = nn.dNN(x_train_d, y_train_d, 'linear', .1, 8)  # training dense nn on full training set 

idealmod_c = nn.cNN(x_train_c, y_train_c, 'relu', .0001, 5, 2, 1)  # trainin convolutional nn on full training set

loss_d, finalaccuracy_d = idealmod_d.evaluate(x_test_d, y_test_d)
loss_c, finalaccuracy_c = idealmod_c.evaluate(x_test_c, y_test_c)

print("Final loss and accuracy of Dense Neural Network: "+str(loss_d)+", "+str(finalaccuracy_d))
print("Final loss and accuracy of Convolutional Neural Network: "+str(loss_c)+", "+str(finalaccuracy_c))










