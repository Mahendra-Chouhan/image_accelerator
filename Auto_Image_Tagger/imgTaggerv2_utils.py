"""
@Author : TeJas.Lotankar
"""

# imports

import streamlit as st
import time

#Duct Tape
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import warnings
warnings.filterwarnings("ignore")

import keras
from keras import applications
from keras.applications import inception_v3
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import seaborn as sns
import pandas as pd

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Duct tape
# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True




# @st.cache(show_spinner=False, suppress_st_warning=True)
def read_img_from_path(img_path, img_size=(224,224)):
	'''
	Reading and preprocessing image from dir as per requirments of different CNN network architecture
	Prams:
		img_path (String) : path to image file
		img_size (tuple-(x,y)) :desired size of output image
		for_squeeze (Boolean) : If True, it'll preprocess for squeezeNet Architecture using 
								--keras.applications.imagenet_utils.preprocess_input--
								else process for other architectures used using 
								--keras.applications.inception_v3.preprocess_input--
	'''
	img = load_img(img_path, target_size=img_size) # Reading image
	img = img_to_array(img) # converting img to array
	img = np.expand_dims(img, axis=0) # Dimension by making the shape (1, inputShape[0], inputShape[1], 3)
	
	img = imagenet_utils.preprocess_input(img)

	return img


# @st.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation = True)
def read_img_object(img_obj, img_size=(224,224), is_imagenet=False):
	'''
	Reading and preprocessing image from OpenCV object as per requirments of different CNN network architecture
	Prams:
		img_object (CpenCV Object) : Pass OpenCV object
		img_size (tuple-(x,y)) :desired size of output image
		for_squeeze (Boolean) : If True, it'll preprocess for squeezeNet Architecture using 
								--keras.applications.imagenet_utils.preprocess_input--
								else process for other architectures used using 
								--keras.applications.inception_v3.preprocess_input--
	'''
	img = cv2.resize(img_obj, img_size) # Reading image
	# img = img_to_array(img) # converting img to array
	img = np.expand_dims(img, axis=0) # Dimension by making the shape (1, inputShape[0], inputShape[1], 3)
	
	if is_imagenet:
		img = inception_v3.preprocess_input(img)
	else:
		img = img/255.
	# img = imagenet_utils.preprocess_input(img)

	return img


@st.cache(show_spinner=True, suppress_st_warning=False, allow_output_mutation = True)
def get_model(model_name = "mobilenet_v2.MobileNetV2"):
	model_exe = "global trained_model; trained_model = applications.{}(include_top=True, weights='imagenet', classes=1000)".format(model_name)
	exec(model_exe)

	# print(type(model))
	return trained_model

@st.cache(show_spinner=True, suppress_st_warning=False, allow_output_mutation = True)
def get_fashion_model(model_name = "MobileNetV2"):
	model_list = {
		"MobileNetV2" : "Auto_Image_Tagger/FashionWeights/FashionData200_MobileNetV2_40eps_71pct_V2.h5",
		"InceptionResNetV2" : "Auto_Image_Tagger/FashionWeights/FashionData300_InceptionResNetV2_40eps_83pct_V4.h5"
	}
	model_loaded = load_model(model_list[model_name], custom_objects={
	'Adam': lambda **kwargs: hvd.DistributedOptimizer(keras.optimizers.Adam(**kwargs))
})
	return model_loaded



def img_tagger(img_object, model_arch):

	list_224 = ['mobilenet_v2.MobileNetV2',
				'vgg16.VGG16',
				'vgg19.VGG19',
				'resnet.ResNet50',
				'resnet.ResNet50',
				'resnet.ResNet50',
				'resnet_v2.ResNet50V2',
				'resnet_v2.ResNet101V2',
				'resnet_v2.ResNet152V2',
				'nasnet.NASNetMobile',
				'densenet.DenseNet121',
				'densenet.DenseNet169',
				'densenet.DenseNet201']
	
	list_299 = ['inception_resnet_v2.InceptionResNetV2',
				'inception_v3.InceptionV3',
				'xception.Xception'
	]

	if model_arch in list_224:
		img_size = (224, 224)
	elif model_arch in list_299:
		img_size = (299, 299)

	img = read_img_object(img_object, img_size, True)
	model = get_model(model_arch)

	global graph
	graph = tf.compat.v1.get_default_graph()
	
	preds = model.predict(img)
	preds_decode = imagenet_utils.decode_predictions(preds)

	return preds_decode, model



def plot_imgnet_results(resDf):
	lbl = []
	probs = []
	for (i, (imagenetID, label, prob_val)) in enumerate(resDf[0]):
		lbl.append(label)
		probs.append(prob_val*100)

	f, ax = plt.subplots(figsize=(7,4))

	sns.set_color_codes("pastel")
	bplt = sns.barplot(lbl, probs)
	
	for p in bplt.patches:
		bplt.text(p.get_x()+p.get_width()/2.,
			p.get_height() +2 ,
            '{:1.2f}%'.format(p.get_height()),
            ha="center")
	
	sns.despine(top=True, right=True, left=True,  bottom=False)
	ax.set_ylabel('')
	ax.set_yticks([])
	plt.xticks(rotation = 10)
	
	st.subheader("Results")
	st.pyplot()



def img_tagger_fashion(img_object, model_arch):
	
	img = read_img_object(img_object, (128,128), False)
	model = get_fashion_model(model_arch)

	preds = model.predict(img)

	# reading class mapping
	with open("D:/WorkPlace/ImageAnalyticsACC/Auto_Image_Tagger/FashionWeights/FashionData_31_ClassMapping.pkl", "rb") as cls_file:
		class_mapping = pickle.load(cls_file)


	# Matching class labels and predictions
	get_label = lambda ind : list( class_mapping.keys() )[ list(class_mapping.values()).index( ind ) ]
	pred_list=[ [get_label(indx), preds[0][indx]*100] for indx in range(0,5)]
	pred_df = pd.DataFrame(pred_list, columns=['label', 'confidance'])

	return pred_df, model

def plot_fashion_result(resDF):

	f, ax = plt.subplots(figsize=(7,4))

	sns.set_color_codes("pastel")
	bplt = sns.barplot('label', 'confidance', data=resDF)
	for p in bplt.patches:
		bplt.text(p.get_x()+p.get_width()/2.,
			p.get_height()  ,
            '{:1.2f}%'.format(p.get_height()),
            ha="center")
	
	sns.despine(top=True, right=True, left=True,  bottom=False)
	ax.set_ylabel('')
	ax.set_yticks([])
	plt.xticks(rotation = 10)

	st.pyplot()
	






def viz_filters(trained_model, img_object, is_imagenet=False):
	'''
	//TODO
	'''
	# Checking model is passed as object or path to model and loading in memory
	# @st.cache(show_spinner=True, suppress_st_warning=True, allow_output_mutation = True)
	def get_model_viz(trained_model):
		if type(trained_model)==str:
			model = load_model(trained_model)
		else:
			model=trained_model

		return model

	model = get_model_viz(trained_model)

	layer_names = []
	layer_outputs = [] # Getting output of layers
	count = 0

	# Getting only top 3 conv layers
	for layer in model.layers[2:]:
		if "conv" in layer.name:
			count+=1
			layer_names.append(layer.name)
			layer_outputs.append(layer.output)
			if count==3:
				break


	activation_model = Model(inputs=model.input, outputs=layer_outputs) # Model for getting activations
	images_per_row = 8 # change according to filters in CNN layers

	img_sz = ( (model.layers[0]).input_shape[0][1], (model.layers[0]).input_shape[0][2] )
	activations = activation_model.predict(read_img_object(img_object, img_sz, is_imagenet))

	for layer_name, activation in zip(layer_names, activations):
		# Building plots with layer mapping
		with st.spinner("Loading {} Layer".format(layer_name)):
			time.sleep(3)
		# st.success("{} Loaded".format(layer_name))

		n_features = activation.shape[-1] # Number of features in the feature map
		
		size = activation.shape[1] #The feature map has shape (1, size, size, n_features).
		n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
		display_grid = np.zeros((size * n_cols, images_per_row * size))
		
		for col in range(n_cols): # Tiles each filter into a big horizontal grid
			for row in range(images_per_row):
				channel_image = activation[0,
												 :, :,
												 col * images_per_row + row]
				channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
				channel_image /= channel_image.std()
				channel_image *= 64
				channel_image += 128
				channel_image = np.clip(channel_image, 0, 255).astype('uint8')
				display_grid[col * size : (col + 1) * size, # Displays the grid
							 row * size : (row + 1) * size] = channel_image

		scale = 1. / size
		plt.figure(figsize=(scale * display_grid.shape[1],
							scale * display_grid.shape[0]))
		plt.title(layer_name)
		plt.grid(False)
		st.subheader(layer_name)
		# st.subheader("You are visualizing {} layer".format(layer_name))
		plt.imshow(display_grid, aspect='auto', cmap='viridis')
		st.pyplot()

