from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, MaxPooling1D
from keras.layers import BatchNormalization
import keras


initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2)

model = Sequential()

#conv1
model.add(ZeroPadding2D((2,2),input_shape=(40,500,1)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#maxpool1
model.add(MaxPooling2D((5,1), strides=(5,1)))
model.add(Dropout(0.50))
#conv2

model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#maxpool2
model.add(MaxPooling2D((2,1), strides=(2,1)))
model.add(Dropout(0.50))

#conv3
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#maxpool3
model.add(MaxPooling2D((2,1), strides=(2,1)))
model.add(Dropout(0.50))

#conv4
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#maxpool4
model.add(MaxPooling2D((2,1), strides=(2,1)))
model.add(Dropout(0.50))

#stacking(reshaping)
model.add(Reshape((500, 16)))

#temporal squeeing
model.add(MaxPooling1D((500), strides=(1)))
model.add(Dropout(0.50))

#fully connected layers
model.add(Flatten())
model.add(Dense(196, activation='sigmoid'))
model.add(Dropout(0.50))
model.add(Dense(2, activation='softmax', name='predictions'))



model.summary()
