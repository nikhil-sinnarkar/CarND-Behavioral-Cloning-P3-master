import numpy as np
import csv
import sklearn
import cv2

X_train = []
Y_train = []
csv_data = []

# read the data in csv file
with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		csv_data.append(line)
	
def generator(csv_data, batch_size = 32):
	data_length = len(csv_data)
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
			center_image = cv2.resize(center_image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
			center_image = center_image[30:70,:]
			center_angle = data[3]
			images.append(center_image)
			steering.append(center_angle)
		
		# preprocess the images here
		
		X_train = np.array(images)
		Y_train = np.array(steering)
		yield sklearn.utils.shuffle(X_train, Y_train)
		

gen_instance = generator(csv_data)
		
while(input()=='y'):
	X_train_batch, Y_train_batch = next(gen_instance)
	print(X_train_batch.shape)
	cv2.imshow('image', X_train_batch[0])
	cv2.waitKey(0)