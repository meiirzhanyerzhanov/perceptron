import numpy as np
import math
from scipy.special import expit
import random

in_dim = 785 # input dimension
out_dim = 10 # number of classes (0-9)
eta = 0.01 # Learning rate. You might try different rates (e.g. 0.001, 0.01, 0.1) to maximize the accuracy
epoch = 5


def Weight_update(feature, label, weight_i2o):
    c = np.dot(feature, weight_i2o)
    # predicted_label = c.index(max(c))
    arr = []
    for i in range(10):
        if ( i == int(label)):
            arr.append(1)
        else :
            arr.append(0)
    pred = np.argmax(c)

    for i in range(len(c)):
        if (pred == i):
            c[i] = 1
        else :
            c[i] = 0
    feature_transposed = []
    feature_transposed.append(feature)
    np_transposed = np.array(feature_transposed)
    array_lab = []
    array_lab.append(arr-c)
    weight_upd = weight_i2o + eta*np_transposed.transpose()*array_lab
    return weight_upd
    ##
	#Update the weights for a train feature.
		# Inputs:
			# feature: feature vector (ndarray) of a data point with 785 dimensions. Here, the feature represents a handwritten digit 
			         # of a 28x28 pixel grayscale image, which is flattened into a 785-dimensional vector (include bias)
			# label: Actual label of the train feature 
			# weight_i2o: current weights with shape (in_dim x out_dim) from input (feature vector) to output (digit number 0-9)
		# Return: updated weight
	##
	#"*** YOUR CODE HERE ***"


def get_predictions(dataset, weight_i2o):
    b = np.dot(dataset, weight_i2o)
    arr = []
    for i in range(0, len(b)):
        pred = np.argmax(b[i])
        arr.append(pred)
    print(arr)
    return arr
    #"""
	#Calculates the predicted label for each feature in dataset.
		# Inputs:
			# dataset: a set of feature vectors with shape  
			# weight_i2o: current weights with shape (in_dim x out_dim)
		# Return: list (or ndarray) of predicted labels from given dataset
	#"""
	#"*** YOUR CODE HERE ***"


def train(train_set, labels, weight_i2o):
    #"""
	#Train the perceptron until convergence.
	# Inputs:
		# train_set: training set (ndarray) with shape (number of data points x in_dim)
		# labels: list (or ndarray) of actual labels from training set
		# weight_i2o:
	# Return: the weights for the entire training set
	#"""
    for i in range(0, train_set.shape[0]):        
        weight_i2o = Weight_update(train_set[i, :], labels[i], weight_i2o)        
    return weight_i2o