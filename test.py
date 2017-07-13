import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend
from keras.models import load_model

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

x_test, y_test = getDataSet("data/test/", imgWidth, imgHeight)
x_test = x_test.reshape(x_test.shape[0], imgWidth, imgHeight, 1)

model = load_model('cnn.h5')

result = model.predict(x_test)

print(np.argsort(result))
