import numpy as np
import math
from scipy.special import expit
import random
in_dim = 785 # input dimension
out_dim = 10 # number of classes (0-9)
eta = 0.01 # Learning rate. You might try different rates (e.g. 0.001, 0.01, 0.1) to maximize the accuracy

def Weight_update(feature, label, weight_i2o):


    #Update the weights for a train feature.
        # Inputs:
            # feature: feature vector (ndarray) of a data point with 785 dimensions. Here, the feature represents a handwritten digit
                     # of a 28x28 pixel grayscale image, which is flattened into a 785-dimensional vector (include bias)
            # label: Actual label of the train feature
            # weight_i2o: current weights with shape (in_dim x out_dim) from input (feature vector) to output (digit number 0-9)
        # Return: updated weight
    ##

    ggg = get_predictions(feature, weight_i2o)
    # creating array of two 10 dimensional y(x) and t(x).
    tx = []
    for i in range(10):
        if (i == int(label)):
            tx.append(1)
        else:
            tx.append(0)
    yx = []
    for i in range(10):
        if (i == int(ggg[0])):
            yx.append(1)
        else:
            yx.append(0)
    tx_np = np.array(tx)
    yx_np = np.array(yx)
    transpose = []
    transpose.append(tx_np - yx_np)
    np_transpose = np.array(transpose)

    feature_arr = []
    feature_arr.append(feature)
    np_feature = np.array(feature_arr)
    # print(np_feature.shape)
    # print(np_transpose.transpose().shape)


    # Calculating weight according to the equation (2)
    ggg2 = np_feature.transpose() * np_transpose

    np_ggg2 = np.array(ggg2)
    weight_i2o = weight_i2o + np_ggg2 * eta

    return weight_i2o
    

def get_predictions(dataset, weight_i2o):




    #"""
    #Calculates the predicted label for each feature in dataset.
        # Inputs:
            # dataset: a set of feature vectors with shape
            # weight_i2o: current weights with shape (in_dim x out_dim)
        # Return: list (or ndarray) of predicted labels from given dataset
    #"""
    arr_dataset = np.array(dataset)
    arr_weight = np.array(weight_i2o)
    pred = []
    score = np.dot(arr_dataset, arr_weight) # calculating score according to the formula

    for i in range(int(dataset.size/785)): # predicting digit for each dataset
       if(len(score)==10): # this case for dataset with shape (785, )
           value = np.argmax(score)
       else: # this case for dataset with shape (n, 785), n>1
           value = np.argmax(score[i, :]) # getting index with the highest value
       pred.append(value) # appending to array predicted value
    return pred


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
