# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:22:15 2020

@author: Student
"""


import matplotlib.pyplot as plt
from mnist import MNIST
import random

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