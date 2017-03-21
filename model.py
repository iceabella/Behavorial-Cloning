# NOTE: for this file all training and validation data needs to be in the same image folder. Multiple csv files are still used.

import csv
import cv2
import numpy as np

# tunable parameters
BATCH_SIZE = 512
correction = 0.15

# use data?
Basic_data = True
Xtra_data = False
CC_data = True
Udacity_data = False

# directories for data
directory_basic = '/home/carnd/Project3/data/'
directory_xtra = '/home/carnd/Project3/data_XTRAdifficultParts/data/'
directory_CC = '/home/carnd/Project3/data_CC2laps/'
directory_udacity = '/home/carnd/Project3/data_udacity/data/'

lines = []
image_names = []
angles = []

# LOAD DATA FOR TRAINING AND VALIDATION
if Basic_data:
    with open(directory_basic + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if Xtra_data:
    with open(directory_xtra + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if CC_data:
    with open(directory_CC + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if Udacity_data:
    with open(directory_udacity + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

# extract the image paths for center, left and right images
# also extract/create steering angle data
for line in lines:
    # add center image path and measurement
    name = directory_basic+'IMG/'+line[0].split('/')[-1]
    center_angle = float(line[3])
    image_names.append(name)
    angles.append(center_angle)

    # add left image path and adjusted steering measurement
    name = directory_basic+'IMG/'+line[1].split('/')[-1]    	
    left_angle = center_angle + correction
    image_names.append(name)
    angles.append(left_angle)

    # add right image path and adjusted steering measurement
    name = directory_basic+'IMG/'+line[2].split('/')[-1]    		
    right_angle = center_angle - correction
    image_names.append(name)
    angles.append(right_angle)      

from sklearn.model_selection import train_test_split
# split into training and validation set
train_images, validation_images, train_angles, validation_angles = train_test_split(image_names,angles,test_size=0.2,random_state=42)

import sklearn

# use generator to be able to train on big amount of data
def generator(image_ref, image_angles, batch_size=BATCH_SIZE):
    num_samples = len(image_ref)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(image_ref,image_angles)
        for offset in range(0, num_samples, batch_size):
            batch_imageRef = image_ref[offset:offset+batch_size]
            batch_imageAngles = image_angles[offset:offset+batch_size]

            images = []
            for batch_sample in batch_imageRef:               
                image = cv2.imread(batch_sample)
                assert image is not None
                images.append(image)

            X_train = np.array(images)
            y_train = np.array(batch_imageAngles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_images, train_angles, batch_size=BATCH_SIZE)
validation_generator = generator(validation_images, validation_angles, batch_size=BATCH_SIZE)

# DEFINE MODEL

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# adding lambda layer for normalization
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
# adding cropping
model.add(Cropping2D(cropping=((50,20),(1,1))))
# NVIDIA layout
model.add(Convolution2D(24,5,5,subsample=(2, 2),activation='relu')) #output size (43,157,24)
model.add(Convolution2D(36,5,5,subsample=(2, 2),activation='relu')) #output size (20,77,36)
model.add(Convolution2D(48,5,5,subsample=(2, 2),activation='relu')) #output size (8,37,48)
model.add(Convolution2D(64,3,3,subsample=(1, 1),activation='relu')) #output size (6,35,64)
model.add(Convolution2D(64,3,3,subsample=(1, 1),activation='relu')) #output size (4,33,64)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# train and validate model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_angles), validation_data=validation_generator, nb_val_samples=len(validation_angles), nb_epoch=10)

# save model
model.save('model.h5')
