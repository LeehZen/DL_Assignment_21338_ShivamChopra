import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
#from sklearn.model_selection import train_test_split
# using fashion_mnist dataset
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np

# load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocessing the images

# check shape of data
_ ,height, width = np.ravel(x_train.shape)

# reshaping data, to fit in dense layers
x_train = x_train.reshape(-1, height*width)
x_test = x_test.reshape(-1, height*width)

#print(x_train.shape)

# Preprocessing the targets

# using one-hot-encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model
dense_model = Sequential([
    Dense(128, activation = 'relu', input_shape = (height*width,)),
    Dense(64, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

#compiling the model
dense_model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# training the model
dense_model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, batch_size = 128)

# evaluation 
loss,accuracy = dense_model.evaluate(x_test, y_test)
print("Loss: ",loss,"\nAccuracy:", accuracy)