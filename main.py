# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:22:15 2020

@author: Student
"""
import os
import PIL
import cv2
import glob
from tkinter import *
from PIL import ImageGrab, Image, ImageDraw
import numpy as np
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import pandas as pd

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


# loads mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshapes to be [samples][width][height][channels]
# makes it suitable to for training in our CNN
# all images are in gray so the pixels array is set to be 1
# 28 represents the length and width of the images
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]


# load the model created from running the train_model.py file
model = load_model('mnist.h5')



def clear_widget():
    global cv
    cv.delete("all")

def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

def Recognize_Digit():
    print("test")
    global image_number
    predictions = []
    percentage = []
    filename = f"image_{image_number}.png"
    widget = cv

    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    ImageGrab.grab().crop((x,y,x1,y1)).save(filename)

    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        roi = th[y-top:y+h+bottom, x-left:x+w+right]
        img = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
        img = img.reshape(1,28,28,1)
        img = img / 255.0
        pred = model.predict([img])[0]
        final_pred = np.argmax(pred)
        data = str(final_pred) + '  ' + str(int(max(pred)*100)) + "%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255,0,0)
        thinkness = 1
        cv2.putText(image, data, (x,y-5), font, fontScale, color, thickness)
    
    cv2.imshow('image', image)
    cv2.waitkey(0)













# GUI
root = Tk()
root.resizable(0,0)
root.title("Digit Recognition App")

lastx, lasty = None, None
image_number = 0

cv = Canvas(root, width=640, height=480, bg="white")
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

cv.bind('<Button-1>', activate_event)

btn_save = Button(text="Recognize Digit", command = clear_widget)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text="Clear Widget", command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

root.mainloop()

            
