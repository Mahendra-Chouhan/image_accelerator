'''
Building streamlit code around Auto Image Tagger.
Auto Image Tagger will take single image as input and returns top 5
classifiaction results.
'''

#Duct Tape
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import cv2
from PIL import Image
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time

from Auto_Image_Tagger.imgTaggerv2_utils import img_tagger, viz_filters, plot_imgnet_results
from Auto_Image_Tagger.imgTaggerv2_utils import img_tagger_fashion, plot_fashion_result
from Auto_Image_Tagger.imgTaggerv2_info import get_info_autoTagger

import warnings
warnings.filterwarnings("ignore")

#Duct Tape
# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True



# ------------------- main_functionality -------------------------------

def main_app():

	model_arch_list = {
			"MobileNetV2" : "mobilenet_v2.MobileNetV2",
			"InceptionResNetV2" : "inception_resnet_v2.InceptionResNetV2",
			"InceptionV3" : "inception_v3.InceptionV3",
			"VGG16" : "vgg16.VGG16",
			"VGG19" : "vgg19.VGG19",
			"Xception" : "xception.Xception",
			"ResNet50" : "resnet.ResNet50",
			"ResNet101" : "resnet.ResNet50",
			"ResNet152" : "resnet.ResNet50",
			"ResNet50V2" : "resnet_v2.ResNet50V2",
			"ResNet101V2" : "resnet_v2.ResNet101V2",
			"ResNet152V2" : "resnet_v2.ResNet152V2",
			"NASNetMobile" : "nasnet.NASNetMobile",
			"DenseNet121" : "densenet.DenseNet121",
			"DenseNet169" : "densenet.DenseNet169",
			"DenseNet201" : "densenet.DenseNet201",
	}


	dataSelect = st.radio("Select dataset", 
		("ImageNet", "FashionData"))

	if dataSelect == "ImageNet":

		inp_img = st.file_uploader("Upload your image, (jpg, png)", type=['jpg','png', 'jpeg'])

		model_arch_sel = st.selectbox('Select model architecture:',list(model_arch_list.keys()))
		model_arch = model_arch_list[model_arch_sel]

		if inp_img:
			img_pil = Image.open(inp_img)
			img_cv2 = np.array(img_pil)
			st.image(img_pil, caption = "Your image", use_column_width=True)
			st.markdown("---")

			img_result, loaded_model = img_tagger(img_cv2, model_arch)

			with st.spinner("Loading Trained Layers"):
				viz_filters(loaded_model, img_cv2, True)

			plot_imgnet_results(img_result)

	elif dataSelect == "FashionData":

		inp_img = st.file_uploader("Upload your image, (jpg, png)", type=['jpg','png', 'jpeg'])

		model_arch_sel = st.selectbox('Select model architecture:',
			("MobileNetV2", "InceptionResNetV2"))

		if inp_img:
			img_pil = Image.open(inp_img)
			img_cv2 = np.array(img_pil)
			st.image(img_pil, caption = "Your image", use_column_width=True)
			st.markdown("---")

			img_result, loaded_model = img_tagger_fashion(img_cv2, model_arch_sel)

			# with st.spinner("Loading Trained Layers"):
			# 	viz_filters(loaded_model, img_cv2, False)

			plot_fashion_result(img_result)



#-------------------------- Front-page ----------------------------


def autoTagger_main_app():
	st.title("Auto image tagger")
	st.markdown("---")
	
	prime_selection = st.sidebar.radio("Please Select following option:",
		("Information", "Auto_Image_Tagger"))

	if prime_selection == "Information":
		get_info_autoTagger()
	elif prime_selection == "Auto_Image_Tagger":
		main_app()