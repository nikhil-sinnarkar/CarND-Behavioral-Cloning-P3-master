# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model architecture

My model architecture consist of 2 Convolutional layer followed by 3 fully connected layer. The kernel size for both the Convolutional layer is 5x5 and filter depth is 24 and 36. The number of neurons for the three fully connected layers are 50, 10 and 1.

```
model.add(Convolution2D(24, 5, 5, subsample = (2,2), input_shape=img.shape, activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation="relu"))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

I have used RELU activations to make the model nonlinear. Also I used a Keras Lambda layer to normalize the data before feeding it to the Convolutional layer.  

```
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=img.shape))
```


#### 2. Attempts to reduce overfitting in the model

To reduce overfitting I have used data augmentation. I have flipped the images captured from the center camera and added them to the orignal data set which is used for training the network.

I trained and validated the model on different data sets to ensure that the model was not overfitting. From the initial data set I took 20% of the data and used it as a validation data. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I used the data from all the 3 cameras i.e. left, center and right to train the model. The left and right camera images were used as recovery data to bring back the car to center of the road. (More details in the next section)


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used a trial and error method to derive my model architecture. 

I started out with the Nvidia architecture which has 3 Convolutional and 4 fully connected layer. I used the default subsample which is (1,1). This architecture seemed a bit overkill for my purpose but still I went along with it just to see how it performed. I trained it for 3 epochs. The training and validation loss were nearly the same. The initial model which I created was about 160 MB in size (size of model.h5). Obviously the model was very big and was trying to extract detailed features from the images. I took a test run with the model and the car drove off the road at the very beginning. 

In my second attempt I removed the 3rd convolutional layer from the previous model and trained for 3 epochs. The training and validation loss were again close to each other but the size of the model reduced to 73 MB (still very big). I again took a test run in the simulator and the car went a bit further compared to previous run. This was a slight improvement hence I further removed 1 fully connected layer in my third attempt, changed the subsample to (2,2) in both the Conv layers and trained again for 3 epoch. This time around I got the size of the model.h5 file down to just 2 MB. I took a test run in the simulator and the car drove way further the before. It fell off the road near the 1st left curve. As my training and validation losses were nearly the same so it was not the case of over fitting. 

So, what was the case then! Well, the car should have taken the left turn when it came to the 1st left curve which it did but the angle was not enough to keep the car on the road. This was because till this piont I had trained the network with only the images from center camera and the network had learned to drive the car mostly in a straingt line, it couldn't steer the car enough to keep it on the road. I needed to balance out the training data. For this I added the images from the left and the right camera (with a small correction value for steering) to the initial data set. I used a positive correction value for the left camera images and negative correction values for right camera images.

```
left_angle = float(data[3]) + left_correction
right_angle = float(data[3]) + right_correction
```

Now whenever the center camera sees an image which is similar to the image from the left camera it tries to steer the car towards right and vice versa. I trained my model again with this data and took a test run in the simulator. This time the car drove all the way through the track without going off the road. 


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        											| 
|:---------------------:|:---------------------------------------------------------------------:| 
| Convolution   		| 24 filters, 5x5 kernel, 2x2 stride, valid padding, RELU Activation	| 
| Convolution   		| 36 filters, 5x5 kernel, 2x2 stride, valid padding, RELU Activation	| 
| Flatten				|   																	|
| Fully connected		| outputs 50															|
| Fully connected		| outputs 10															|
| Fully connected		| outputs 1																|



#### 3. Creation of the Training Set & Training Process

For trainig my model I used the data set provided by Udacity. I used some preprocessing and data augmentation techniques on this data set. 
* The size of images in the data set was 160x320x3. One sample image is shown below.

	![image1](./writeup_images/image1.jpg)

* I cropped out some of the upper and lower portion of the image as it was not useful for the training process. The cropped image is shown below.

	![cropped_image](./writeup_images/cropped_image.jpg)

* After cropping the images I flipped them and added left and right camera images to the data set.

	![image2](./writeup_images/image2.jpg)

* When flipping the images I also change the sign of steering angle (+ to - & - to +). This data is equivalent to drive the car in opposite direction on the track.

* I also scaled down the images before feeding it to the Model.

```
image = cv2.resize(image, None, fx=0.25, fy=0.4, interpolation = cv2.INTER_CUBIC)
```

After the augmentation process I had 32,144 data points.


I finally randomly shuffled the data set and fed it to the model. I kept 20% of the initial data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trined the model for 3 epochs as after this there wasn't any considerable reduction in the loss and the model performance was good in the simulator. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I ran the model in the simulator and recorded the video which you can watch [here]:(https://github.com/nikhil-sinnarkar/CarND-Behavioral-Cloning-P3-master/blob/master/video.mp4).
