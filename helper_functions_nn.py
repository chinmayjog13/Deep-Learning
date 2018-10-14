# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 05:12:59 2018

@author: chint
"""

#Assumed data to be (input layer size, number of samples)
# Network architecture is passed as a list- eg- [input neurons, layer1_neurons, ..., output neurons]

import numpy as np
import pickle
from datetime import datetime
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x)), x

def relu(x):
    return np.maximum(0, x), x

def leaky_relu(x):
    return np.maximum(0.2*x, x), x

def tanh(x):
    return np.tanh(x), x

def sigmoid_back(dA, cache):
    Z = cache
    s, _ = sigmoid(Z)
    return dA * s * (1-s)

def relu_back(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def tanh_back(dA, cache):
    Z = cache
    t, _ = tanh(Z)
    return dA * (1 - t*t)   # Derivative of tanh

def leaky_relu_back(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<0] = 0.2
    return dZ

def activation(x, use_activation):
    assert use_activation in ['sigmoid', 'relu', 'tanh', 'leaky_relu', None]
    if use_activation == 'sigmoid':
        out = sigmoid(x)
    
    elif use_activation == 'relu':
        out = relu(x)
    
    elif use_activation == 'tanh':
        out = tanh(x)
    
    elif use_activation == 'leaky_relu':
        out = leaky_relu(x)
        
    else:
        out = x
    
    return out

def init_parameters(layer_dims):
    np.random.seed(1)
    L = len(layer_dims)
    parameters = {}
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters

def forward(A, W, b):
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def forward_block(A_prev, W, b, use_activation):
    Z, lin_cache = forward(A_prev, W, b)
    A, act_cache = activation(Z, str(use_activation))
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (lin_cache, act_cache)
    
    return A, cache

def model_forward(X, parameters, activations):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        A, cache = forward_block(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activations[l-1])
        caches.append(cache)
        #print(parameters['W' + str(l)].shape, parameters['b' + str(l)].shape)
    
    Al, cache = forward_block(A, parameters['W' + str(L)], parameters['b' + str(L)], activations[L-1])
    caches.append(cache)
    #print(parameters['W' + str(L)].shape, parameters['b' + str(L)].shape)
    
    assert(Al.shape == (1,X.shape[1]))
    return Al, caches

def compute_cost(Al, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum( np.multiply(Y, np.log(Al)) + np.multiply((1-Y), np.log(1-Al)))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

def back(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.matmul(W.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def back_block(dA, cache, use_activation):
    lin_cache, act_cache = cache
    
    if str(use_activation) == 'sigmoid':
        dZ = sigmoid_back(dA, act_cache)
        dA_prev, dW, db = back(dZ, lin_cache)
    
    elif str(use_activation) == 'relu':
        dZ = relu_back(dA, act_cache)
        dA_prev, dW, db = back(dZ, lin_cache)
    
    elif str(use_activation) == 'tanh':
        dZ = tanh_back(dA, act_cache)
        dA_prev, dW, db = back(dZ, lin_cache)
        
    elif str(use_activation) == 'leaky_relu':
        dZ = leaky_relu_back(dA, act_cache)
        dA_prev, dW, db = back(dZ, lin_cache)
        
    return dA_prev, dW, db

def model_back(Al, Y, caches, activations):
    grads = {}
    L = len(caches)
    Y = Y.reshape(Al.shape)
    
    dAl = -(np.divide(Y, Al) - np.divide(1-Y, 1-Al))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = back_block(dAl, current_cache, activations[L-1])
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = back_block(grads["dA" + str(l+1)], current_cache, activations[l])
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]
    
    return parameters

def save_model(parameters):
    model = open('./model_{}.pkl'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), 'wb')
    pickle.dump(parameters, model)
    model.close()

def get_model(name):
    model = open(name, 'rb')
    parameters = pickle.load(model)
    model.close()
    return parameters

def train(X, Y, layer_dims, activations, learning_rate=0.0075, num_iter=2500, print_cost=False, plot_cost=False):
    np.random.seed(1)
    costs = []
    
    parameters = init_parameters(layer_dims)
    for i in range(0, num_iter):
        Al, caches = model_forward(X, parameters, activations)
        cost = compute_cost(Al, Y)
        grads = model_back(Al, Y, caches, activations)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print('Cost after iteration {}: {}'.format(i, cost))
            costs.append(cost)
    if plot_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    return parameters

def predict(X, Y, parameters, activations):
    m = X.shape[1]      #Number of examples
    p = np.zeros((1,m))
    
    probs, caches = model_forward(X, parameters, activations)
    for i in range(0, probs.shape[1]):
        if probs[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print("Accuracy: "  + str(np.sum((p == Y)/m)))
        
    return p
