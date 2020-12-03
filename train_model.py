
# load and format the mnist dataset

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
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

def simple_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
    
# define the larger model
def larger_model():
	# create model
	model = Sequential()
   # optimizer = tf.keras.optimizers.Adam(0.001)
   # optmizer = tf.keras.optimizers.Adam(0.001)
    #optimizer.learning_rate.assign(0.01)
    # creates the first layer of the network which expects the inputted images
    # this layer has 30 feature maps and will compute the output of the dot 
    # product between the weights and region of the input volume of each neuron
    # essentially this takes a dot product of ionput values that are near each other
    # in a region and assciates that value in 1 of the 30 feature maps
	model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    # pooling layer with size 2x2
    # this layer takes the most prominent values from the convolutional layer
    # and skips over the others to improve processing time and reduce data sie
	model.add(MaxPooling2D())
    # creates a second convulational layer to reduce the featues down to 15 with
    # sizes 3x3 
	model.add(Conv2D(15, (3, 3), activation='relu'))
    # pools the features in second layer by skipping over some less prominent values
	model.add(MaxPooling2D())
     # dropout layer that exludes 20% of the nuerons
    # this prevents over specilizing our model with too many layers by reducing complexity
	model.add(Dropout(0.2))
    # converts output to a vector
	model.add(Flatten())
    # fully connected layer which takes in the flattened output and then determines the value to output 
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# build the model

def threeLayer():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(20, (4, 4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(12, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
model = larger_model()
#model = simple_model()
#model = threeLayer()
#small_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
#scores = small_model.evaluate(X_test, y_test, verbose=0)
#print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
# Fit the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Model CNN Error: %.2f%%" % (100-scores[1]*100))
#three_layer.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
#scores = three_layer.evaluate(X_test, y_test, verbose=0)
#print("Three layer CNN Error: %.2f%%" % (100-scores[1]*100))