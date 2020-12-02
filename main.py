# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:22:15 2020

@author: Student
"""

import matplotlib.pyplot as plt
from mnist import MNIST
import random

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

# this function may end up belonging to network class
def gradientDescent(training_images, training_labels, batch_size, iterations) :
    # the training set is fed through the network in smaller batches to speed training process
    for i in range(iterations):
        for k in range(training_images.length):
            batch_x, batch_y = training_images[k:k+batch_size], 
            training_labels[k:k+batch_size];
            
mndata = MNIST('data')

images, labels = mndata.load_training()
# or
#images, labels = mndata.load_testing()

index = random.randrange(0, len(images))  # choose an index ;-)
print(mndata.display(images[index]))
#def main() :
#    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
   
   # %matplotlib inline # Only use this if using iPython
#    image_index = 7777 # You may select anything up to 60,000
#    print(y_train[image_index]) # The label is 8
#    plt.imshow(x_train[image_index], cmap='Greys')

