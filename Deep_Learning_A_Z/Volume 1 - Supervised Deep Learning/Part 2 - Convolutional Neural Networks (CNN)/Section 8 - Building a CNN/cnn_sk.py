#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:56:26 2018

@author: sudhir
"""
# =============================================================================
# Convolution neural network
# Part 1 Building cnn
# Importing the keras libraries and packages
# =============================================================================
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D, Flatten, Dense

# Initialising the CNN
classifier = Sequential()

# Step1 - convolution 
classifier.add(Convolution2D(32,3,3, input_shape =(64,64,3), activation='relu'))

# Step2 - MaxPooling
classifier.add(MaxPool2D(pool_size =(2,2)))

# Second convolution layer
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPool2D(pool_size =(2,2)))

# Step3 - Flattening
classifier.add(Flatten())

# Step 4 -Fullconnection
classifier.add(Dense(output_dim= 128, activation= 'relu')) 
classifier.add(Dense(output_dim= 1, activation= 'sigmoid')) 

# Compiling  the CNN
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics =['accuracy'])

# =============================================================================
# Part2 - Fitting the CNN to the image
# Image augumentation
# =============================================================================
from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                        'dataset/training_set',
                        target_size=(64, 64),
                        batch_size=32,
                        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=3,
        validation_data=test_set,
        validation_steps=2000)

# =============================================================================
# Part3 - Making new prediction
# =============================================================================
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] ==1:
    prediction = 'dog'
else:
    prediction = 'cat'
