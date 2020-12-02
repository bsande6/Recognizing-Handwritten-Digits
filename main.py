# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:22:15 2020

@author: Student
"""

import matplotlib.pyplot as plt
from mnist import MNIST
import random
# i dont know why importing this class no longer works
from network.py import Network


mndata = MNIST('data')

# could convert this to one tuple containing image and label
training_images, training_labels = mndata.load_training()
training_images, training_labels = mndata.load_testing()

# just prints out a random image to check read in
index = random.randrange(0, len(training_images))  # choose an index ;-)
print(mndata.display(training_images[index]))
#def main() :
    
learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5
n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)

# creates network object with 3 layers, number of layers can be changed if desired
net = Network([n_hidden1, n_hidden2, n_hidden3])
net.gradientDescent(training_images, training_labels, batch_size, n_iterations, learning_rate)


            
