# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:22:15 2020

@author: Student
"""

from tkinter import *
import tkinter as tk
#import win32gui
from PIL import ImageGrab, Image
import numpy as np
from app import App
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical

def predict_digit(img):
    # resize image to 28x28 pixels
    img = img.resize((28,28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    # predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = xtrain.reshape((xtrain.shape[0], 28, 28, 1))
xtest = xtest.reshape((xtest.shape[0], 28, 28, 1))
input_shape = (28, 28, 1)
xtrain = xtrain / 255
xtest = xtest / 255
ytrain = to_categorical(ytrain, 10)
ytest = to_categorical(ytest, 10)
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain = xtrain / 255
xtest = xtest / 255


# load the model created from running the train_model.py file
model = load_model('mnist.h5')


app = App()
mainloop()


            
