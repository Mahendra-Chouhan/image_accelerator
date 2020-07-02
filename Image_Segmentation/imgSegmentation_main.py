"""
@Author : TeJas.Lotankar

Description:
------------
	Main file for image segmentation calls.
"""

# imports

import streamlit as st

from Image_Segmentation.imgSegmentation_utils import get_img_segment
from Image_Segmentation.imgSegmentation_info import get_info_segment

from gluoncv import model_zoo, data, utils
import mxnet as mx

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


# ----------------------------- Streamlit_Utils ----------------------------

def get_image():
	
	CONTAINER_DIR = "./container/imgSegmentation/"
	if not os.path.isdir(CONTAINER_DIR):
		os.makedirs(CONTAINER_DIR)

	temp_img_path = "./container/imgSegmentation/temp_img_imgSeg.jpg"
	img_bytes = st.file_uploader("Upload your image, (jpg, png)", type=['jpg','png', 'jpeg'])
	if img_bytes:
		img_pil = Image.open(img_bytes).save(temp_img_path)
		return temp_img_path



# ----------------------------- Main_Functionality --------------------------


def main_app():
	# inp_img = st.file_uploader("Upload your image, (jpg, png)", type=['jpg','png', 'jpeg'])
	# st.title("Image Segmentation")
	st.markdown("---")

	st.header("Upload image:")
	sel_way = st.radio("How you want to upload image?", 
		("Local_Disk", "URL"))

	if sel_way == "Local_Disk":
		inp_img = get_image()
	elif sel_way == "URL":
		inp_img = st.text_input("Image URL:")

	if inp_img:
		# img_pil = Image.open(inp_img)
		# img_cv2 = np.expand_dims(np.array(img_pil), axis=0)
		# im = mx.image.imdecode(np.asarray(img_pil))
		# st.write(img_cv2.shape)


		model_arch = st.selectbox('Select network architecture',
			(	'mask_rcnn_resnet50_v1b_coco',
				'mask_rcnn_resnet18_v1b_coco',
				'mask_rcnn_resnet101_v1d_coco'))

		threshold = st.slider("Threshold:", 0.10, 1.0, 0.70, 0.05)

		class_check = False #st.checkbox("Select limited classes?")

		cls_lst=[]

		# if class_check:
		# 	net = model_zoo.get_model(model_arch, pretrained=True)
		# 	model_cls = net.classes
		# 	cls_lst = st.multiselect("Select classes you want to include:",model_cls)


		ax = get_img_segment(inp_img, model_arch, threshold, class_check, cls_lst)
		st.pyplot()










#-------------------------- Front-page ----------------------------

# Adding logo
# st.sidebar.markdown("[![Yash_Technologies](https://www.yash.com/wp-content/themes/Yash/images/yash-logo.webp)](https://www.yash.com/)")
# st.sidebar.markdown("---")

def imgSegment_main_app():
	st.title("Image Segmentation")
	prime_selection = st.sidebar.radio("Please Select following option:",
		("Information", "Image_Segmentation"))

	if prime_selection == "Information":
		get_info_segment()
	elif prime_selection == "Image_Segmentation":
		main_app()

# Documentation
# st.sidebar.info("Please read Information for more details about this project before starting.\
# 	For custom training follow folder structure of training images")

# st.sidebar.title("Contribution")
# st.sidebar.markdown("![Twitter](https://img.icons8.com/color/48/000000/twitter.png)"+"[TeJas Lotankar](https://twitter.com/tejas_radax)")
