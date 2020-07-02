"""
@Author : TeJas.Lotankar

Description:
------------
	Main function for object detection utils and calls.
"""


# imports

import streamlit as st

from Object_Detection.objDetect_utils import get_object_detection
from Object_Detection.objDetect_info import get_info_objDetect

from gluoncv import model_zoo, data, utils
import mxnet as mx

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


# ----------------------------- Streamlit_Utils ----------------------------

def get_image():
	
	CONTAINER_DIR = "./container/objDetectection/"
	if not os.path.isdir(CONTAINER_DIR):
		os.makedirs(CONTAINER_DIR)

	temp_img_path = "./container/objDetectection/temp_img_objDetect.jpg"
	img_bytes = st.file_uploader("Upload your image, (jpg, png)", type=['jpg','png', 'jpeg'])
	if img_bytes:
		img_pil = Image.open(img_bytes).save(temp_img_path)
		return temp_img_path


# ----------------------------- Main_Functionality --------------------------


def main_app():

	st.header("Upload image:")
	sel_way = st.radio("How you want to upload image?", 
		("Local_Disk", "URL"))

	if sel_way == "Local_Disk":
		inp_img = get_image()
	elif sel_way == "URL":
		inp_img = st.text_input("Image URL:")

	if inp_img:

		model_arch = st.selectbox('Select network architecture',
			(	'ssd_512_resnet50_v1_voc',
				'ssd_512_mobilenet1.0_voc',
				'yolo3_darknet53_voc',
				'yolo3_mobilenet1.0_voc',
				'ssd_512_resnet50_v1_coco',
				'ssd_512_mobilenet1.0_coco',
				'yolo3_darknet53_coco',
				'yolo3_mobilenet1.0_coco'))

		threshold = st.slider("Threshold:", 0.10, 1.0, 0.70, 0.05)

		class_check = False #st.checkbox("Select limited classes?")

		cls_lst=[]

		# if class_check:
		# 	net = model_zoo.get_model(model_arch, pretrained=True)
		# 	model_cls = net.classes
		# 	cls_lst = st.multiselect("Select classes you want to include:",model_cls)


		ax = get_object_detection(inp_img, model_arch, threshold, class_check, cls_lst)
		st.pyplot()



#-------------------------- Front-page ----------------------------


def objDetect_main_app():
	st.title("Object Detection")
	st.markdown("---")
	
	prime_selection = st.sidebar.radio("Please Select following option:",
		("Information", "Object_Detection"))

	if prime_selection == "Information":
		get_info_objDetect()
	elif prime_selection == "Object_Detection":
		main_app()
