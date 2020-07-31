import numpy as np
import math
from scipy.special import expit
import random

in_dim = 785 # input dimension
out_dim = 10 # number of classes (0-9)
eta = 0.01 # Learning rate. You might try different rates (e.g. 0.001, 0.01, 0.1) to maximize the accuracy

def Weight_update(feature, label, weight_i2o):

        label_vector = np.zeros(10)
        label_vector[int(label)] = 1.0
        y_t = np.dot(feature, weight_i2o)
        y_t[y_t > 0] = 1.0
        y_t[y_t <= 0] = 0.0
        weight_i2o += eta * np.outer(feature, np.subtract(label_vector, y_t))
        return weight_i2o


def get_predictions(dataset, weight_i2o):
    vecs = np.dot(dataset, weight_i2o)
    print(weight_i2o.shape)
    print(dataset.shape)
    print(vecs.shape)
    correct_num = 0
    arr = []
    print(dataset.shape[0])
    for i in range(dataset.shape[0]):
        pred = np.argmax(vecs[i, :])
        arr.append(pred)
    return arr


def train(train_set, labels, weight_i2o):
        for i in range(0, train_set.shape[0]):
            weight_i2o = Weight_update(train_set[i, :], labels[i], weight_i2o)

        return weight_i2o


# In[2]:



import math
from scipy.special import expit
from sklearn.metrics import confusion_matrix, accuracy_score
import random
# import Perceptron_Lib


num_epochs = 5

def load_data(file_name):
    data_file = np.loadtxt(file_name, delimiter=',')
    dataset = np.insert(data_file[:, np.arange(1, in_dim)]/255, 0, 1, axis=1)
    data_labels = data_file[:, 0]
    return dataset, data_labels

# Initialize weight matrix with random weights
weight = np.random.uniform(-0.05,0.05,(in_dim,out_dim))

# Load Training and Test Sets :
print("\nLoading Training Set")
train_data, train_labels = load_data('train.csv')
print("\nLoading Test Set\n")
test_data, test_labels = load_data('test.csv')

arr_train_acc = []
arr_test_acc = [] 

for i in range(1, num_epochs+1):
    
    pred_train_labels = get_predictions(train_data, weight)  # Test network on training set and get training accuracy
    curr_accu = accuracy_score(train_labels, pred_train_labels)

    print("Epoch " + str(i) + " :\tTraining Set Accuracy = " + str(curr_accu))

    pred_test_labels = get_predictions(test_data, weight)  # Test network on test set and get accuracy on test set
    test_accu = accuracy_score(test_labels, pred_test_labels)
    print("\t\tTest Set Accuracy = " + str(test_accu))

    weight = train(train_data, train_labels, weight)    # Train the network
    arr_train_acc.append(curr_accu)
    arr_test_acc.append(test_accu)

# Test network on test set and get test accuracy
pred_test_labels = get_predictions(test_data, weight)  # Test network on test set and get accuracy on test set
test_accu = accuracy_score(test_labels, pred_test_labels)

# Confusion Matrix
print("\t\tFinal Accuracy = " + str(test_accu) + "\n\nConfusion Matrix :\n")
print(confusion_matrix(test_labels, pred_test_labels))
print("\n")

