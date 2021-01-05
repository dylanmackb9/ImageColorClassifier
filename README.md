# Image Color Classifier

Classifying yellow and blue images based on their color through Deep Learning algorithms. Using a Dense Neural Network and Convolutional Neural Network to compare computation time, hyper parameters, and final loss and accuracy. 

Created 1000 yellow and blue images using python PIL package, and utilized feature extraction to create a feature to recognize pixel value from RGBA value. Split into a training and test set. 

Used TensorFlow 2 and Keras to create a two models:


Dense NN: Created a single hidden layer fully connected sequential neural network functioning 	as a multilayer perceptron since it uses a linear activation function. 

Convolutional NN: Created 6 layer convolution neural network with two convolution layers and 	   	maximum pooling layers, finishing with a flattening layer, a fully connected, and a softmax output 	layer. The convolution layers used filter dimensionality output size of 2 and 4, respectively. The 		convolution layers also used a kernel size of 3 for both layers. The max pool layer used a pool 		size of 2. The final fully connected layer had 64 nodes giving a max of 36,928 parameters  		tuned.

	
K-fold Cross Validation was used to evaluate model performance on hyper-parameters:  activation function, learning rate, number of training epochs, kernel, and stride. A simple Grid Search algorithm was used to optimize hyper-parameters. 
	
The purpose of this was to practice deep learning techniques with simple images, and to compare traditional neural networks with convolutional neural networks in performance and computation time. 
