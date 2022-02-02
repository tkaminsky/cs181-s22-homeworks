#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

x_test = np.arange(0, 12, .1)

# Kernel Function
def K(x1, x2, tau):
    return np.exp(-1 * np.linalg.norm(x1 - x2) ** 2 / tau)

# Finds the array of indices of the k nearest neighbors to point x
def point_knn(x, k, tau):
    # Indexes of the k closest points
    close_indices = np.array(range(k))
    # Distances of the k closest points (initialized to first k points)
    close_dists = np.array([K(x, x_train[i], tau) for i in close_indices])
    
    for i,el in enumerate(x_train):
        # Finds the difference between el and x
        dist = K(x, el, tau)
        # Finds the furthest known point from x
        min_index = np.argmin(close_dists, axis=0)
        # If el is closer, have it replace it
        if (close_dists[min_index] <= dist and not i in close_indices):
            close_indices[min_index] = i
            close_dists[min_index] = dist
            
    return close_indices
        
        
    

def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    predictions = np.zeros(len(x_test))

    for i,x in enumerate(x_test):
        # Indices of the k nearest neighbors 
        indices = point_knn(x, k, tau)
        
        predictions[i] = np.sum([y_train[j] for j in indices]) / k
    
    return predictions
    
    


def plot_knn_preds(k):
    plt.xlim([0, 12])
    plt.ylim([0,3])
    
    y_test = predict_knn(k=k)
    
    plt.scatter(x_train, y_train, label = "training data", color = 'black')
    plt.plot(x_test, y_test, label = "predictions using k = " + str(k))

    plt.legend()
    plt.title("KNN Predictions with k = " + str(k))
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for k in (1, 3, len(x_train)-1, len(x_train)):
    plot_knn_preds(k)