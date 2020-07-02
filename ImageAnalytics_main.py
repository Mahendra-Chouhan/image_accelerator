"""
@Author : TeJas.Lotankar

Description:
------------
	Image Analytics Acc main function for calling all apps.
"""


# imports

import streamlit as st

from ImageAnalytics_info import get_info_main
from Pre_Processing_Acc.preProcessing_main_app import pp_main_app
from Custom_clf.imgClf_main_app import img_clf_main
from Object_Detection.objDetect_main import objDetect_main_app
from Image_Segmentation.imgSegmentation_main import imgSegment_main_app
from Auto_Image_Tagger.imgTaggerv2_main import autoTagger_main_app
from Pose_Estimation.poseEstimation_main import poseEstimation_main_app
from Action_Recognition.actionRecognition_main import actionRecognition_main_app

from Defect_Detection.defectDetection_main import defectDetection_main_app





# Duct tape
#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True


#-------------------------- Front-page ----------------------------

# Adding logo
st.sidebar.markdown("[![Yash_Technologies](https://www.yash.com/wp-content/themes/Yash/images/yash-logo.webp)](https://www.yash.com/)")
st.sidebar.markdown("---")

st.sidebar.header("Image Analytics Accelerator")

apps_list = {
	"About" : "get_info_main()",
	"Pre-Processing_Accelerator" : "pp_main_app()",
	"Custom_Classification" : "img_clf_main()",
	"Auto_Image_Tagger" : "autoTagger_main_app()",
	"Object_Detection" : "objDetect_main_app()",
	"Defect_Detection" : "defectDetection_main_app()",
	"Image_Segmentation" : "imgSegment_main_app()",
	"Pose_Estimation" : "poseEstimation_main_app()",
	"Action_Recognition" : "actionRecognition_main_app()"
}

# app_selection = st.sidebar.selectbox("Please Select appliction:",
# 	(
# 		"About",
# 		"Pre-Processing_Accelerator",
# 		"Custom_Classification",
# 		"Object_Detection",
# 		"Image_Segmentation"
# 	))

app_selection = st.sidebar.selectbox("Please Select appliction:",
	list(apps_list.keys()))
st.sidebar.markdown("---")


exec(apps_list[app_selection])

# Documentation
# st.sidebar.info("For more information please go through about page.")

