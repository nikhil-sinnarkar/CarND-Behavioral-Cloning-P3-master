import numpy as np
import csv
import sklearn
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda

X_train = []
Y_train = []
csv_data = []
left_correction = 0.28
right_correction = -0.28

# read the data in csv file
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None) # skip the header
	for line in reader:
		csv_data.append(line)

def preprocess(image):
	# change the color space from BGR to RGB
	# crop the image and downscale them
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image[60:140,:]
	image = cv2.resize(image, None, fx=0.25, fy=0.4, interpolation = cv2.INTER_CUBIC)
	return image


# calculate the shape of images after preprocessing to pass the shape to Keras Convolutional layer
img_path = './data/IMG/'+csv_data[0][0].split('/')[-1]
img = cv2.imread(img_path)
img = preprocess(img)
np.array(img)
print("Image shape passed to Keras Input is ",img.shape)

# split the data from CSV file into training and validation data
from sklearn.model_selection import train_test_split
train_data, validation_data = train_test_split(csv_data, test_size=0.2)


def generator(csv_data, batch_size = 28):
	data_length = len(csv_data)
	while 1:
		# sklearn.utils.shuffle(csv_data)
		for offset in range(0, data_length, batch_size):
			# split the data into batches
			batch_data = csv_data[offset:offset+batch_size]
			# list to store images and steering angle for each batch
			images = []
			steering = []
			
			for data in batch_data:
				image_path = './data/IMG/'+data[0].split('/')[-1]
				center_image = cv2.imread(image_path)
				center_image = preprocess(center_image)
				center_angle = float(data[3])
				images.append(center_image)
				steering.append(center_angle)
				
				image_path = './data/IMG/'+data[1].split('/')[-1]
				left_image = cv2.imread(image_path)
				left_image = preprocess(left_image)
				left_angle = float(data[3]) + left_correction
				images.append(left_image)
				steering.append(left_angle)

				image_path = './data/IMG/'+data[2].split('/')[-1]
				right_image = cv2.imread(image_path)
				right_image = preprocess(right_image)
				right_angle = float(data[3]) + right_correction
				images.append(right_image)
				steering.append(right_angle)

				flipped_image = cv2.flip(center_image, 1)
				flipped_angle = -float(data[3])
				images.append(flipped_image)
				steering.append(flipped_angle)				
			
						
			X_train = np.array(images)
			Y_train = np.array(steering)
			yield sklearn.utils.shuffle(X_train, Y_train)
		

gen_instance = generator(train_data)
validation_generator = generator(validation_data, batch_size=32)
		
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=img.shape))
model.add(Convolution2D(24, 5, 5, subsample = (2,2), input_shape=img.shape, activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation="relu"))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(gen_instance, samples_per_epoch=len(train_data)*4, verbose=1, validation_data=validation_generator, nb_val_samples=len(validation_data), nb_epoch=3)
model.save('drive_model.h5')
