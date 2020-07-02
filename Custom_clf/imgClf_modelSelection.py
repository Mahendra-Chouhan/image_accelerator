'''
@Author : TeJas.Lotankar

This class will contain methods for calling CNN models architectures

Methods
-----------
load_inceptionResNetV2(model="inceptionResNetV2", class_num, img_size=(299, 299, 3))
	This method defines and initiates models from user choise,
	gets number of classes to be predicted,
	image size and number of epochs is needed.

'''

# imports
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pprint import pprint
from tqdm import tqdm
import os
import cv2
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import Conv2D
from keras.optimizers import Adam,SGD,RMSprop,Adagrad
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.callbacks import ReduceLROnPlateau
from keras import callbacks
from keras.layers import BatchNormalization
import time

from keras.applications import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.mobilenet_v2 import MobileNetV2

import streamlit as st

def load_model(class_num, model="inceptionnet", img_size=(299, 299, 3), loss_function='categorical_crossentropy', optimizer='Adam'):
	'''
	Description
	------------
	This method defines and initiates models from user choise,
	gets number of classes to be predicted,
	image size and number of epochs is needed.


	Parameters
	------------
	model : str
		Model name to be used for training and classification
		(Default "inceptionResNetV2")

	class_num : int
		Number of classes for image classification. (Default None)

	img_size : tuple (299, 299, 3)
		Size of image to be processed into while training for classification.
		(Default set for InceptionResNetV2 i.e. (299, 299, 3))

	loss_function : keras loss function
		//TODO

	optimizer : keras optimizer
		//TODO



	Returns
	---------
	Compiled keras model with weights

	'''

	MODELS = {
			"xceptionnet": Xception,
			"inceptionnet": InceptionResNetV2,
			"mobilenet": MobileNetV2
			# "squeezenet": squeezeNet
	} # Making Key:Value pair of models for various calls

	# ensure a valid model name was supplied via command line argument
	if model not in MODELS.keys():
		raise AssertionError("The model argument should be of following:\n"
			"- xceptionnet\n"
			"- inceptionnet\n"
			"- mobilenet\n"
			"- Not implemented now ----squeezenet----\n")

	model_call = MODELS[model]

	# Defining base model
	base_model = model_call(
		include_top=False, 
		weights='imagenet', 
		input_shape=img_size)

	input_tensor = Input(shape=img_size)
	bn = BatchNormalization()(input_tensor)
	x = base_model(bn)
	x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
	x = Flatten()(x)
	x = Dropout(0.5)(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	output = Dense(class_num, activation='softmax')(x)
	model_out = Model(input_tensor, output)

	# Freezing layers of base model.
	# For fine tuning change the number of layer to be freezed.
	for layer in base_model.layers:
		layer.trainable=False

	# Optimisers
	opt = Adam(lr=1e-2)


	#Compile

	model_out.compile(
		loss=loss_function,
		optimizer=optimizer,
		metrics=['accuracy'])

	return model_out

