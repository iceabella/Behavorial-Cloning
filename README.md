# Behavorial-Cloning
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.png "Center image"
[image2]: ./examples/right.png "Right Image"
[image3]: ./examples/left.png "Left Image"
[image4]: ./examples/opposite.png "Opposite direction"
[image5]: ./examples/flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network 
* README.md summarizing the results
* video1.mp4 video showing my vehicle driving autonomously around track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the [NVIDIA model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), since they used their model for a similar problem I thought that it was an appropriate starting point. The model consists of five convolutional layers with filter sizes of 5x5 and 3x3 and a depth between 24 and 64 (model.py lines 116-120). Detailed information about the network architechture can be seen later.

The model includes RELU layers to introduce nonlinearity (code line 116), the data is normalized in the model using a Keras lambda layer (code line 112) and also cropped (code line 114). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 28-75). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 128).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving both from clockwise and counter-clockwise direction from the first track. Additionally I used both the right and the left camera images as well.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make a step by step approach to see how the model performance was increasing/decreasing by different additions.

My first step was to use a convolutional neural network model similar to the LeNet architechture with a normalization layer. I started by evaluating my model on 2 laps of data from track 1 driving in clockwise direction. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I thought the validation and test errors was quite good but the model was not performing well on the test track. 

When I was looking at my test and validation errors again I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added more data by first augment the data by flipping the images and steering angles. The model was still overfittin so I was thereafter also adding the left and right camera images with an adjusted steering measurement. This additionally helped improving steering bias and the vehicle to know how to get back to the middle of the lane. This made my car pass the bridge for the first time!

Since the vehicle was able to get quite far on the track (i.e. predict the steering angle ok) I decided to try to crop the images to get the model to focus on what is important. This improved the driving and the vehicle was able to drive almost the whole test track!

Since the model was still overfitting I collected more data by driving 2 laps counter-clockwise on track 1. The model got a quite small error on training and validation, but when I tried it on the test track it didn't perform that well, why I chose to use the NVIDIA model. I used this model since it has been used to the same task before.

To be sure about what to use for the NVIDIA model I used more or less the same incrementing complexity as for the LeNet model. I also added a generator function to be able to use more data for training and validation. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolutional neural network with the following layers and layer sizes:

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 160x320x3 RGB image   			|
| Normalization		| outputs 160x320x3 image			| 
| Cropping	     	| cropping pixels: 50 top and 20 bottom pixels, outputs 90x318x3|
| Convolution 5x5 filter| 2x2 stride, relu activation, valid padding, outputs 43x157x24		|
| Convolution 5x5 filter| 2x2 stride, relu activation, valid padding, outputs 20x77x36		|
| Convolution 5x5 filter| 2x2 stride, relu activation, valid padding, outputs 8x37x48		|
| Convolution 3x3 filter| 1x1 stride, relu activation, valid padding, outputs 6x35x64		|
| Convolution 3x3 filter| 1x1 stride, relu activation, valid padding, outputs 4x33x64		|
| Fully connected	| outputs 100								|
| Fully connected	| outputs 50								|
| Fully connected	| outputs 10								|
| Output		| outputs 1								|



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I additionally used the left and right camera images so that the vehicle would learn to drive back to the middle if getting too close to the edges of the road. These shows what a left and a right image looks like:

![alt text][image3]
![alt text][image2]

I also used recorded data from going in the opposite direction on track one.
![alt text][image4]

After the collection process, I had 15938 number of data points. I then preprocessed this data by cropping the images.

To note is that I used only the keyboard to steer the vehicle during data collection, why the data most likely is not optimal for training.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by that the validation loss started increasing again after epoch 4. I used an adam optimizer so that manually training the learning rate wasn't necessary.


#### Additional information

The model has some drawbacks and can be improved further, some interesting findings has also been seen.

Since the data used was only recorded from the first track the model is not general enough to be able to drive on the second track with a satisfying result. Something interesting that was seen is that by training the model only on data collected by driving one lap counter-clockwise on track 2 the model is able to run around track 1 as well (not as smooth, but possible). The model also gets more general and is able to drive a part of track 2 as well. Since track 2 is more difficult and has more turns it can be seen that the car is more likely to turn even on track 1, why the behaviour is worse than the used model (turns a lot when the road is straight). This shows that we would need a lot of varying data to get a generalized model that can drive on different kinds of roads. Also some additional preprocessing technique could possibly improve the performance.

