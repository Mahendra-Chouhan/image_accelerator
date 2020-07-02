"""
@Author : TeJas.Lotankar

Description:
------------
	Main function for pose estimation utils and calls.
"""


# imports

import streamlit as st

from Pose_Estimation.poseEstimation_utils import get_pose_estimation
from Pose_Estimation.poseEstimation_info import get_info_poseEstimation

from gluoncv import model_zoo, data, utils
import mxnet as mx

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


# ----------------------------- Streamlit_Utils ----------------------------

def get_image():
	
	CONTAINER_DIR = "./container/poseEstimation/"
	if not os.path.isdir(CONTAINER_DIR):
		os.makedirs(CONTAINER_DIR)

	temp_img_path = "./container//poseEstimation/temp_img_poseEst.jpg"
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

		model_arch_detect = st.selectbox('Select detection network:',
			(	
				'yolo3_darknet53_coco',
				'yolo3_mobilenet1.0_coco',
				'yolo3_mobilenet1.0_voc',
				'yolo3_darknet53_voc',
				'ssd_512_resnet50_v1_voc',
				'ssd_512_mobilenet1.0_voc',
				'ssd_512_resnet50_v1_coco',
				'ssd_512_mobilenet1.0_coco',
				))

		model_arch_pose = st.selectbox("Select Pose Estimation network:",
			(
				"alpha_pose_resnet101_v1b_coco",
				"simple_pose_resnet18_v1b",
				"simple_pose_resnet50_v1b",
				"simple_pose_resnet101_v1b",
				"simple_pose_resnet152_v1b"
				))

		box_threshold = st.slider("Box Threshold:", 0.10, 1.0, 0.50, 0.05)
		keypoint_threshold = st.slider("Keypoint Threshold:", 0.10, 1.0, 0.20, 0.05)

		ax = get_pose_estimation(inp_img, 
			model_arch_detect, model_arch_pose, 
			box_threshold, box_threshold)

		st.pyplot()



#-------------------------- Front-page ----------------------------


def poseEstimation_main_app():
	st.title("Pose Estimation")
	st.markdown("---")
	
	prime_selection = st.sidebar.radio("Please Select following option:",
		("Information", "Pose_Estimation"))

	if prime_selection == "Information":
		get_info_poseEstimation()
	elif prime_selection == "Pose_Estimation":
		main_app()