import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras import initializers
import numpy as np
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = f"Spoiled_Classifier_v1-{time.time()}"

tb = TensorBoard(log_dir=f"logdir/{NAME}")


X = np.load("features.npy")
y = np.load("labels.npy")
y = to_categorical(y)

X = X/255.0

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))

model.add(Conv2D(64,(4,4)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))


#last layer
model.add(Dense(25))
model.add(Activation('softmax'))

model.compile(loss = "categorical_crossentropy",optimizer= 'adam',metrics= ["accuracy"])

model.fit(X,y,batch_size=16,validation_split=.1, epochs=15, )

# for last param callbacks = [tb]
