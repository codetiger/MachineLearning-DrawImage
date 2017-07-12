import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend

import numpy as np
import os
from PIL import Image

imgWidth, imgHeight = 28, 28

def getFiles(folderPath):
	files = []
	fileNumbers = []
	for f in os.listdir(folderPath):
		if os.path.isfile(os.path.join(folderPath, f)) and not f.startswith("."):
			files.append(f)
			fileNumber = os.path.splitext(f)[0]
			fileNumbers.append(int(fileNumber))

	return files, np.array(fileNumbers)

def getResizedImageData(folderPath, files, width, height):
	imgData = []
	for f in files:
		img = Image.open(os.path.join(folderPath, f))
		# print("Image:" + f + " Mode: " + img.mode)
		img = img.convert(mode='P', palette='ADAPTIVE')
		img.thumbnail((width, height), Image.BILINEAR)
		data = np.asarray( img, dtype="int8" )
		imgData.append(data)
	return np.array(imgData)

def getDataSet(folderPath, imgW, imgH):
	files, fileNumbers = getFiles(folderPath)
	imgData = getResizedImageData(folderPath, files, imgW, imgH)
	return imgData, fileNumbers

x_train, y_train = getDataSet("data/train/", imgWidth, imgHeight)
x_test, y_test = getDataSet("data/test/", imgWidth, imgHeight)

x_train = x_train.reshape(x_train.shape[0], imgWidth, imgHeight, 1)
x_test = x_test.reshape(x_test.shape[0], imgWidth, imgHeight, 1)

batch_size = 128
num_classes = len(x_train)
epochs = 12

input_shape = (imgWidth, imgHeight, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('cnn.h5')