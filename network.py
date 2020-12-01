# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:02:20 2020

@author: Student
"""
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # initializes biases and weights to random values
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    
    # parameters that network will update in training process through each batch
    # these functions are how the network learns
    def updateWeights():
        #having a temporary var here removes errors of empty function
        placeholder = 0;
        
    def updateBias() :
        placeholder = 0;
       
    # not sure where this function belongs but we will need it for gradient descent    
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))   