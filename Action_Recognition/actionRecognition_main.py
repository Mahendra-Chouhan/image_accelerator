"""
@Author : TeJas.Lotankar

Description:
------------
	Main function for acion recognition utils and calls.
"""


# imports

import streamlit as st

from Action_Recognition.actionRecognition_utils import get_action_recognition
from Action_Recognition.actionRecognition_info import get_info_actionRecognition

from gluoncv import model_zoo, data, utils
import mxnet as mx

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import os
import pandas as pd


# ----------------------------- Streamlit_Utils ----------------------------

def get_video():
	
	CONTAINER_DIR = "./container/actionRecognition/"
	if not os.path.isdir(CONTAINER_DIR):
		os.makedirs(CONTAINER_DIR)

	temp_vid_path = "./container//actionRecognition/temp_vid_actionRecog.mp4"
	vid_frame = st.file_uploader("Upload your video, (mp4)", type=["mp4"])
	if vid_frame:
		with open(temp_vid_path, "wb") as outFile:
			outFile.write(vid_frame.getbuffer())

	return temp_vid_path



# ----------------------------- Main_Functionality --------------------------


def main_app():

	st.header("Upload video:")
	sel_way = st.radio("How you want to upload video?", 
		("Local_Disk", "URL"))

	if sel_way == "Local_Disk":
		inp_vid = get_video()
	elif sel_way == "URL":
		inp_vid = st.text_input("video URL:")

	if inp_vid:

		model_arch = st.selectbox('Select detection network:',
			(	
				'slowfast_4x16_resnet50_kinetics400',
				'slowfast_8x8_resnet50_kinetics400',
				'slowfast_8x8_resnet101_kinetics400',
				'inceptionv1_kinetics400',
				'inceptionv3_kinetics400',
				'i3d_resnet101_v1_kinetics400',
				'i3d_resnet50_v1_kinetics400',
				'i3d_inceptionv1_kinetics400',
				'i3d_inceptionv3_kinetics400'
				))

		resDF = get_action_recognition(inp_vid, model_arch)

		f, ax = plt.subplots(figsize=(7,4))

		sns.set_color_codes("pastel")
		bplt = sns.barplot(x="class", y="prob", data=resDF)
		
		for p in bplt.patches:
			bplt.text(p.get_x()+p.get_width()/2.,
				p.get_height() +5 ,
	            '{:1.2f}%'.format(p.get_height()),
	            ha="center")
		
		sns.despine(top=True, right=True, left=True,  bottom=False)
		ax.set_ylabel('')
		ax.set_yticks([])
		plt.xticks(rotation = 10)
		
		st.subheader("Results")
		st.pyplot()


#-------------------------- Front-page ----------------------------


def actionRecognition_main_app():
	st.title("Action Recognition")
	st.markdown("---")
	
	prime_selection = st.sidebar.radio("Please Select following option:",
		("Information", "Action_Recognition"))

	if prime_selection == "Information":
		get_info_actionRecognition()
	elif prime_selection == "Action_Recognition":
		main_app()