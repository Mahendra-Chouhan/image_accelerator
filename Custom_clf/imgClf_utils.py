'''
@Author : TeJas.Lotankar

This contains functions required for image classification, data reading, and preprocessing

Methods
-----------
get_train_img (dir_path)
	This method gets directory path from user and read all images and labels

preprocess_img(img_df, img_size=(256,256), batch_size=64)
	This method accepts dataframe['img_dir', 'img_label']
	and returns processed keras.ImageDataGenerator iterators for training.

'''

# imports

import streamlit as st
from PIL import Image

import pickle

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Duct Tape
#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True


def get_train_img(dir_path):
	'''
	Description
	------------
	This method gets directory path from user and read all images and labels
	

	Parameters
	------------
	dir_path : str
		Path to directory containing all training images (Default None)


	Returns 
	------------
	Dataframe containing path for each image and label

	'''
	img_classes = [i for i in tqdm(os.listdir(dir_path))] # walks through given path and gathers name of directory

	# Reading images from dir
	img_dir=[]
	for lbl in tqdm(img_classes):
		for img in os.listdir(dir_path+lbl):
			img_dir.append([dir_path+lbl+'/'+img, lbl])

	# Making Dataframe of all images
	img_dir_df = pd.DataFrame(img_dir, columns=['image_path', 'label'])

	return img_dir_df


def preprocess_img(img_df, img_size, batch_size=8):
	'''
	Description
	------------
	This method accepts dataframe['img_dir', 'img_label']
	and returns processed keras.ImageDataGenerator iterators for training.
	

	Parameters
	-----------
	img_df : DataFrame
		Pandas dataframe having two columns as ['img_dir', 'img_label'] (Default None)

	image_size : tuple (299, 299)
		Image size to be processed with.

	batch_size : int
		Number of images to be processed at a time. (Default to 8)


	Returns
	-------------------
	Three iterators for:
		- training data
		- validation data
		- testing data
	Dictionary of all class indices
	'''

	train_df, test_df = train_test_split(img_df, test_size=0.15) #splitting data

	categories_list = img_df.label.unique()

	# Creating Datagenrators
	train_datagen = ImageDataGenerator(
		rescale=1. / 255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		validation_split=0.10)

	test_datagen = ImageDataGenerator(rescale=1./255)

	# Creating genrators
	train_img_data = train_datagen.flow_from_dataframe(dataframe=train_df, 
		x_col='image_path', y_col='label', 
		batch_size=batch_size, class_mode='categorical', 
		target_size=img_size, subset="training")

	val_img_data = train_datagen.flow_from_dataframe(dataframe=train_df, 
		x_col='image_path', y_col='label', 
		batch_size=batch_size, class_mode='categorical', 
		target_size=img_size, subset="validation")

	test_img_data = test_datagen.flow_from_dataframe(dataframe=test_df, 
		x_col='image_path', y_col='label', 
		batch_size=batch_size, 
		target_size=img_size)

	# Getting class indices for prediction
	class_indices = train_img_data.class_indices
	# print(class_indices)

	return train_img_data, val_img_data, test_img_data, class_indices



def predict_on_model(img_path, model, dest_path, class_indices):
	'''
	Description
	------------
	Reads the images form path, 
	Process images, 
	Predict on model, 
	then images are saved to destination folder

	Parameters
	------------
	img_path : str
		Path to the single image or directory of images for prediction.

	model : keras model
		Trained keras model for prediction of images.

	dest_path : str
		Destination path where images will be stored after prediction.

	class_indices : dictionary
		Dictionary of class indices which contains key-value mapping
		of class labels and indexes.

	'''

	# Getting image size from model imput shape
	img_size = model.input_shape


	# Above function is converted into pytohn Lambda for in memory processing
	proc_img = lambda img: ((cv2.resize( cv2.imread(img, cv2.IMREAD_COLOR), (img_size[1], img_size[2]) ))/255).reshape(1, img_size[1], img_size[2], img_size[3])


	# Reading images if provided path is directory
	if os.path.isdir(img_path):
		img_list = [os.path.join(img_path, i) for i in tqdm(os.listdir(img_path))]
	else:
		img_list = [img_path]

	# Checking for destination path, if not creating one
	if not os.path.isdir(dest_path):
		os.makedirs(dest_path)


	for img in tqdm(img_list):
		img_lbl = list( class_indices.keys() )[ list(class_indices.values()).index( (model.predict( proc_img(img) ) ).argmax()) ]
		img_dest = os.path.join( dest_path, img_lbl+"_"+img.split('/')[-1] )
		cv2.imwrite(img_dest, cv2.imread(img))



def predict_single_image(input_img, model, class_indices,dest_path=None):
	'''
	Description
	------------
	Reads the images form path or object, 
	Process images, 
	Predict on model, 
	then images are saved to destination folder or returned as object.

	Parameters
	------------
	input_img : str or OpenCV image object
		Path to the single image or directory of images for prediction.

	model : keras model
		Trained keras model for prediction of images.

	dest_path : str (Default : None)
		Destination path where images will be stored after prediction.
		If not provided, it'll be set to None and result will not be saved.

	class_indices : dictionary
		Dictionary of class indices which contains key-value mapping
		of class labels and indexes.

	'''

	# Getting image size from model imput shape
	img_size = model.input_shape

	# Above function is converted into python Lambda for in memory processing
	# Processing image from path
	proc_img_from_path = lambda img: ((cv2.resize( cv2.imread(img, cv2.IMREAD_COLOR), (img_size[1], img_size[2]) ))/255).reshape(1, img_size[1], img_size[2], img_size[3])

	# Processing image object
	proc_img_obj = lambda img: ((cv2.resize( img, (img_size[1], img_size[2]) ))/255).reshape(1, img_size[1], img_size[2], img_size[3])

	# Checking input type
	if type(input_img)==str:
		proc_img = proc_img_from_path(input_img)
	else:
		proc_img = proc_img_obj(input_img)


	# predicting on image
	img_lbl = list( class_indices.keys() )[ list(class_indices.values()).index( (model.predict( proc_img ) ).argmax()) ]

	
	if dest_path:
		# Checking for destination path, if not creating one
		if not os.path.isdir(dest_path):
			os.makedirs(dest_path)
		img_dest = os.path.join( dest_path, img_lbl+"_"+img.split('/')[-1] )
		cv2.imwrite(img_dest, cv2.imread(img))


	return img_lbl


def show_taining_data(dir_path):
	'''
	//TODO
	'''

	img_train_df = get_train_img(dir_path)

	st.subheader("Inspecting Training Data")
	if st.button("Show random image from taining data"):
		temp_img = img_train_df.sample(n=1).values.tolist()[0]
		st.image(Image.open(temp_img[0]), caption=temp_img[1], use_column_width=True)





@st.cache(show_spinner=True, suppress_st_warning=False, allow_output_mutation = True)
def get_model(model_path):
	# st.header("Came here...")
	model = load_model(model_path) # Loading model from saved .h5 files
	return model

def show_testing_results(dir_path, trained_model, class_file):
	'''
	//TODO
	'''
	

	model = get_model(trained_model)
	img_train_df = get_train_img(dir_path)
	img_size = model.input_shape
	with open(class_file, "rb") as cls_file:
		class_indices = pickle.load(cls_file)


	proc_img_from_path = lambda img: ((cv2.resize( cv2.imread(img, cv2.IMREAD_COLOR), (img_size[1], img_size[2]) ))/255).reshape(1, img_size[1], img_size[2], img_size[3])

	get_label = lambda ind : list( class_indices.keys() )[ list(class_indices.values()).index( ind ) ]

	st.subheader("Showing results:")
	temp_df = img_train_df.sample(n=1).values.tolist()[0]
	
	img_to_pred = proc_img_from_path(temp_df[0])
	preds = model.predict(img_to_pred)
	# print(preds)
	
	# Matching class labels and predictions
	pred_list=[ [get_label(indx), preds[0][indx]*100] for indx in range(0,len(preds[0]))]
	pred_df = pd.DataFrame(pred_list, columns=['label', 'confidance'])
	
		
	# img_lbl = list( class_indices.keys() )[ list(class_indices.values()).index( (model.predict( img_to_pred ).argmax() )) ]
	

	# Plotting results
	f, ax = plt.subplots(figsize=(12,6))

	plt.imshow(cv2.imread(temp_df[0], 1))
	plt.title("Actual label : {}".format(temp_df[1]))
	plt.xticks([])
	plt.yticks([])
	st.pyplot()

	sns.set_color_codes("pastel")
	bplt = sns.barplot('label', 'confidance', data=pred_df)
	for p in bplt.patches:
		bplt.text(p.get_x()+p.get_width()/2.,
			p.get_height() +2 ,
            '{:1.2f}%'.format(p.get_height()),
            ha="center")
	
	sns.despine(top=True, right=True, left=True,  bottom=False)
	ax.set_ylabel('')
	ax.set_yticks([])
	plt.xticks(rotation = 10)

	st.pyplot()
	
	



















# Testing
# PATH = "./images/"

# a = get_train_img(PATH)
# print(a.head())
# train_df, test_df = train_test_split(a, test_size=0.15)
# print(a.label.unique())
# print("-----------------------")
# print(train_df.label.unique())
# print("-----------------------")
# print(test_df.label.unique())