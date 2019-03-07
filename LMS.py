# This is a library of routines that calculates batch and stochastic gradient descent
# Part of HW2 for CS_5350
# Created by Janaan Lake, February 2019

import numpy as np
from numpy import linalg as LA

def LMS_regression(X, y, gradient_method="stochastic",learning_rate=0.25):
    """
    Calculates the least means square regression for a given dataset.  Plots the losses against the number
    of iterations.
    
    Input:
    -X:  a numpy array of shape [no_samples, no_attributes] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels for the dataset
    -gradient_method:  "stochastic" or "batch"
    -learning_rate:  A float representing the learning rate to be used in calculating the gradient.
    
    Returns:
    -W:  a numpy array of shape [no_attributes] representing the set of weights learned in the algorithm.
    """

    import numpy as np
    from numpy import linalg as LA
    import matplotlib.pyplot as plt
    import random

    W = np.zeros(X.shape[1]) # initialize the weights as zero
    W_update = np.zeros(X.shape[1])
    eps = 0.000001
    convergence = False
    counter = 0
    losses = []
    if gradient_method == "stochastic":
        while not convergence:
            
            counter += 1
            loss = 0.0
            #choose a random example
            rand_sample = random.randint(0,X.shape[0]-1)
            gradient = (y[rand_sample] - (X[rand_sample].dot(W))) * X[rand_sample]
            loss = 0.5 * (y[rand_sample] - (X[rand_sample].dot(W)))**2
            losses.append(loss)
            W_update = W + learning_rate * gradient
            
            #the norm is how we measure convergence
            update_norm = LA.norm(W_update - W)
            if update_norm < eps:
                convergence = True
                print("The final loss on the training data is " + str(loss))
                
            W = W_update
                
    else:
        #batch gradient descent
        while not convergence:
            
            counter += 1
            loss = 0.0
            gradient = 0.0
            for i in range(X.shape[0]):
                loss += 0.5 * (y[i] - (X[i].dot(W)))**2
                gradient += (y[i] - (X[i].dot(W))) * X[i]
            losses.append(loss / X.shape[0])
            W_update = W + learning_rate * gradient / X.shape[0]
            
            update_norm = LA.norm(W_update - W)
            if update_norm < eps:
                convergence = True
                print("The final loss on the training data is " + str(loss / X.shape[0]))
            
            W = W_update
                
    plt.title('Loss')
    plt.xlabel('Number of Iterations')
    plt.plot(losses)
    plt.show()
    
    print("The total number of iterations is " + str(counter))
    return W



def predict(X, y, W):
    """
    Computes the loss of a dataset X and given weight vector W.
    
    Inputs:
    -X:  a numpy array of shape [no_samples, no_attributes] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels for the dataset
    -W:  a numpy array of shape [no_attributes] representing the set of weights to predict the labels.
    
    Returns:
    -loss:  A float representing the average loss of each sample in X.
    """
    
    loss = 0.0
    for i in range(X.shape[0]):
        loss += 0.5 * (y[i] - (X[i].dot(W)))**2
    return loss / X.shape[0]


def load_data(path):
    """
    Loads and processes the concrete data set
    
    Inputs:
    -path:  string representing the path of the file
    
    Returns:
    -X:  a numpy array of shape [no_samples, no_attributes]
    -y:  a numpy array of shape [no_samples] that represents the labels for the dataset X
    """
    
    import numpy as np
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            example = line.strip().split(',')
            if len(example) > 0:
                example = [float(i) for i in example]
                data.append(example)
    X = np.array(data, dtype=np.float64)
    y = X[:,-1]
    X = X[:,:-1]
    
    return X, y


#Test the batch and stochastic gradient descent algorithms on the concrete dataset
X, y = load_data("train.csv")
X_test, y_test = load_data("test.csv")

#stochastic gradient descent:
for lr in (0.1, 0.06, 0.01):
    print("LMS regression for stochastic gradient descent")
    print("The learning rate is " + str(lr))
    W = (LMS_regression(X,y,gradient_method="stochastic", learning_rate=lr))
    print("The weights are :")
    print(W)
    loss = predict(X_test, y_test, W)
    print("The loss on the test data is " + str(loss))
    print("\n")
    
#batch gradient descent:
for lr in (0.5, 0.3, 0.25):
    print("LMS regression for batch gradient descent")
    print("The learning rate is " + str(lr))
    W = (LMS_regression(X,y,gradient_method="batch", learning_rate=lr))
    print("The weights are :")
    print(W)
    loss = predict(X_test, y_test, W)
    print("The loss on the test data is " + str(loss))
    print("\n")

    
print("The correct weights are: ")
alpha = np.dot(np.dot(LA.inv(np.dot(X.T,X)),X.T),y.T)
print(alpha)


