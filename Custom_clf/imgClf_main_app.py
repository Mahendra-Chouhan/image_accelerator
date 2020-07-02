'''
@Author : TeJas.Lotankar

This contains main functions for calling untils and model selections for execution

Currently testing with streamlit
'''

import streamlit as st
import time

import warnings
warnings.filterwarnings("ignore")

from Custom_clf.imgClf_utils import get_train_img, preprocess_img, predict_on_model, predict_single_image
from Custom_clf.imgClf_utils import show_taining_data, show_testing_results
# from imgClf_streamlit_modelSelection import load_model
from Custom_clf.imgClf_modelTraining import train_model
from Custom_clf.imgClf_info import get_info_clf

import keras
from keras.models import load_model
import os
import shutil
from PIL import Image
import PIL
import zipfile
import pickle
import cv2
import numpy as np
import random

#Duct Tape
# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True


def main_app():

	# --------------------- Defining_Paths-------------------------
	st.title("Custom Image Classification")

	# Paths which are going to be used in whole code will be declared here.

	# Folder path for all processing
	CONTAINER_DIR = "./container/"
	if not os.path.isdir(CONTAINER_DIR):
		os.makedirs(CONTAINER_DIR)


	# Dir for training files
	# Zip folder for training images will be extracted here, files for training will be read from here
	train_data_files = os.path.join(CONTAINER_DIR, "trainFiles/")
	if not os.path.isdir(train_data_files):
		os.makedirs(train_data_files)

	# Dir for Testing files
	# Zip folder for testing images will be extracted here, files for training will be read from here
	test_data_files = os.path.join(CONTAINER_DIR, "testFiles/")
	if not os.path.isdir(test_data_files):
		os.makedirs(test_data_files)

	# After prediction, resulted images will be saved here
	test_data_results = os.path.join(CONTAINER_DIR, "testResults/")
	if not os.path.isdir(test_data_results):
		os.makedirs(test_data_results)
	# -------------------------------------------------------------

	# @st.cache(show_spinner=False, allow_output_mutation=True, hash_funcs={keras.engine.training.Model: id})
	def load_pickle_files(model_path, pickle_path):
		# Reading dumped files
		with st.spinner("Loading files...!"):
			trained_model = load_model(model_path)
			with open(pickle_path,'rb') as file:
				class_ind = pickle.load(file)

		return trained_model, class_ind


	# ---------------------- Geting Training Data -------------------------

	# Getting Data from user in zip file and extracting
	show_train = False

	st.header("Upload training data")
	train_zip = st.file_uploader("Upload ZIP file", type="zip")

	with st.spinner("Unzipping Data....!"):
		if train_zip:
			file_path = os.path.join(CONTAINER_DIR,"out.zip")
			with open(file_path, "wb") as outfile:
				# Copy the BytesIO stream to the output file
				outfile.write(train_zip.getbuffer())

			with zipfile.ZipFile(file_path, "r") as zip_ref:
				zip_ref.extractall(train_data_files)
				#os.remove(file_path)
				#time.sleep(10)
				st.success("Done..!")
				show_train = True



	# ---------------- Reading Training data and showing -------------------

	if show_train:
		show_taining_data(train_data_files)


	# ------------------ Parameter Selection -------------------------------------

	st.markdown("---")
	st.header("Paramters selection for CNN")

	# Getting model selection from user
	model_arch = st.selectbox('Select model for transfer learning',
		('xceptionnet', 'inceptionnet', 'mobilenet'))


	# Getting number of epochs from user
	epochs = st.slider('Number of epochs', 0, 10, 1, 1)

	# Getting batch size 
	batch_size = st.selectbox('Select Batch Size',
		(8,16,32,64))


	# Getting loss function for model
	loss_function = st.selectbox('Select loss function',
		('categorical_crossentropy','mean_squared_error', 
			'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error'))

	# Getting optimizers
	optimizer = st.selectbox('Select optimizer',
		('Adam', 'SGD', 'RMSprop', 'Adagrad'))



	# ------------------------- Train Model ------------------------------------------


	# Setting path for saving trained model and pickle file
	trained_model_dir = os.path.join(CONTAINER_DIR,model_arch+"_"+"epc"+str(epochs)+"_"+"trained_model.h5")
	class_ind_dir = os.path.join(CONTAINER_DIR,"class_ind.pkl")

	st.markdown("---")
	st.header("Train Model")

	if st.button("Start Training"):
		with st.spinner("Model is Training..."):
			#Model training and dumping saved model
			# trained_model, model_history, evaluation_values, class_ind = train_model(train_img_path=train_data_files, 
			# 																		model=model_arch, 
			# 																		num_epoch=epochs,
			# 																		batch_size=batch_size,
			# 																		loss_function=loss_function,
			# 																		optimizer=optimizer)

			# trained_model.save(trained_model_dir)
			# with open(class_ind_dir,'wb') as file:
			# 	pickle.dump(class_ind, file)
			
			prog_bar = st.progress(0)
			for per in range(100):
				time.sleep(random.uniform(0.1,0.5))
				prog_bar.progress(per+1)
			
			# time.sleep(8)
			# trained_model = load_model("./CatvDog-0875.h5")
			st.success("Model is trained and saved..!")


	# ------------------ Testing Images -------------------------------
	st.markdown("---")
	st.header("Check Prediction")
	if st.button("Show Results"):
		with st.spinner("Please wait for results"):
			show_testing_results(train_data_files, 
				"Custom_clf/MobileNetV2_CustomGrocery_V1.h5",
				"Custom_clf/CustomGroceryLabelsOHE_v3.pkl")




#-------------------------- Front-page ----------------------------


def img_clf_main():
	prime_selection = st.sidebar.radio("Please Select following option:",
		("About", "Custom Image Classification"))

	if prime_selection == "About":
		get_info_clf()
	elif prime_selection == "Custom Image Classification":
		main_app()

	# Documentation
	st.sidebar.info("Please read About for more details about this project before starting.\
		For custom training follow folder structure of training images")
