import numpy as np

def relu(x):
    return np.maximum(0,x)

def lrelu(x):
    return np.maximum(0.01*x, x)

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def softmax(X):
    A = []
    for i in X:
        A.append(np.exp(i)/np.sum(np.exp(X)))
    return A