import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# load and format the mnist dataset
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

print(ytrain)

# create the model
batch_size = 128
num_classes = 10
epochs = 10
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

# train the model and save the results in mnist.h5
hist = model.fit(xtrain, ytrain, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (xtest, ytest))
print("Model trained successfully")
score = model.evaluate(xtest, ytest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('mnist.h5')
print("Saved the model to mnist.h5")