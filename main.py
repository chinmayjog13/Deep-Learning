# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 01:21:15 2018

@author: chint
"""
# Logistic regression for detecting cat vs non-cat images
import numpy as np
import h5py
import helper_functions_nn as hf
import argparse
from load_data import load_data

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-t', '--train', default=True, type=bool,
                    help="Training mode, Train if true, test if false")
parser.add_argument('-l', '--layer_dims', type=str, required=True,
                    help="Model architecture parameters without input dimensions")
parser.add_argument('-m', '--model', default='',
                    help="Model name if pretrained model")
parser.add_argument('-a', '--activations', type=str, required=True,
                    help='Activation functions for each layer')

args = parser.parse_args()

is_train = args.train
architecture = [int(i) for i in (args.layer_dims).split(',')]
activations = (args.activations).split(',')
print(activations)

train_x, train_y, test_x, test_y, classes = load_data()
train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T

train_x = train_x_flatten/255.0
test_x = test_x_flatten/255.0

layer_dims = [train_x.shape[0]] + architecture

if is_train and args.model == '':
    parameters = hf.train(train_x, train_y, layer_dims, activations, print_cost=True, plot_cost=True)
    hf.save_model(parameters)

if args.model != '':
    parameters = hf.get_model(args.model)

train_acc = hf.predict(train_x, train_y, parameters, activations)
test_acc = hf.predict(test_x, test_y, parameters, activations)