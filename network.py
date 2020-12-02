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
    def updateWeights(self, data, learning_rate):
        for x, y in data:
            # calculate gradient (backpropagation algorithm)
            self.weights = 0
        
    def updateBias(self, data, learning_rate):
        
        self.bias = 0
       
  
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))   
    
        
    def gradientDescent(self, training_images, training_labels, batch_size, iterations, learning_rate) :
        # the training set is fed through the network in smaller batches to speed training process
        length = len(training_images)
        for i in range(iterations):
            for k in range(0, length, batch_size):
                batch_x, batch_y = training_images[k:k+batch_size], 
                training_labels[k:k+batch_size]
                
                for b in [batch_x, batch_y]:
                    self.updateWeights(b, learning_rate)
                    self.updateBias(b, learning_rate)
                
                
                