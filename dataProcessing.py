# run in geoguessr venv


import numpy as np
from PIL import Image
from sklearn.utils import shuffle



# Creating training set of 1000 blue and yellow images to train on 

ex_blue = Image.new("RGBA", (30,30), "blue")  # creating single blue image
ex_yellow = Image.new("RGBA", (30,30), "yellow")  # creating single yellow image 
blueData = np.array(ex_blue)  # turning image into numpy array
yellowData = np.array(ex_yellow)  # turning image into numpy array

data_features_blue = blueData.reshape((1,30,30,4))  # blue training data
data_labels_blue = np.zeros((500,))  # LABELING BLUE O

data_features_yellow = yellowData.reshape((1,30,30,4))  # yellow training data
data_labels_yellow = np.ones((500,))  # LABELING YELLOW 1

for i in range(499):  # creating 500 training examples of each class
	im_blue = np.array(Image.new("RGBA", (30,30), "blue")).reshape((1,30,30,4))  # creating new blue picture data
	im_yellow = np.array(Image.new("RGBA", (30,30), "yellow")).reshape((1,30,30,4))  # creating new yellow picture data

	data_features_blue = np.concatenate((data_features_blue, im_blue), axis=0)  # adding new image to train set
	data_features_yellow = np.concatenate((data_features_yellow, im_yellow), axis=0)  # adding new image to train set 


# Testing 
#print(train_data_features_blue.shape)
#print(train_data_labels_blue.shape)
#print(train_data_features_yellow.shape)
#print(train_data_labels_yellow.shape)

dataset_features = np.concatenate((data_features_blue, data_features_yellow), axis=0)
dataset_labels = np.concatenate((data_labels_blue, data_labels_yellow))

print(dataset_features.shape)
print(dataset_labels.shape)

dataset_features, dataset_labels = shuffle(dataset_features, dataset_labels, random_state=0)


np.save("dataset_features", dataset_features)
np.save("dataset_labels", dataset_labels)






