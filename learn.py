import numpy as np
import csv
import sklearn
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

X_train = []
Y_train = []
csv_data = []

# read the data in csv file
with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		csv_data.append(line)

from sklearn.model_selection import train_test_split
train_data, validation_data = train_test_split(csv_data, test_size=0.2)
	
def generator(csv_data, batch_size = 32):
	data_length = len(csv_data)
	while 1:
		sklearn.utils.shuffle(csv_data)
		for offset in range(0, data_length, batch_size):
			# split the data into batches
			batch_data = csv_data[offset:offset+batch_size]
			# list to store images and steering angle for each batch
			images = []
			steering = []
			
			for data in batch_data:
				image_path = data[0]
				center_image = cv2.imread(image_path)
				# center_image = cv2.resize(center_image, (64,64))
				center_image = cv2.resize(center_image, None, fx=0.36, fy=0.5, interpolation = cv2.INTER_CUBIC)
				center_image = center_image[30:70,:]
				center_angle = float(data[3])
				images.append(center_image)
				steering.append(center_angle)
			
			# preprocess the images here
			
			X_train = np.array(images)
			Y_train = np.array(steering)
			yield sklearn.utils.shuffle(X_train, Y_train)
		

gen_instance = generator(csv_data)
validation_generator = generator(validation_data, batch_size=32)
		
model = Sequential()
model.add(Convolution2D(24, 5, 5, input_shape=(40,115,3), activation="relu"))
model.add(Convolution2D(36, 5, 5, activation="relu"))
model.add(Convolution2D(48, 5, 5, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(gen_instance, samples_per_epoch=len(train_data), verbose=1, validation_data=validation_generator, nb_val_samples=len(validation_data), nb_epoch=3)
model.save('drive_model.h5')
