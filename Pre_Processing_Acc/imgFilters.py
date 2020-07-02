"""
@Author : TeJas.Lotankar

Description:
------------
	Various filters to apply on image for applying changes
		- Blur filter
		- Edge Filter
		- 
		//TODO Add More filters as per necessity

"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

import streamlit as st
from PIL import Image


def blur_filter_img(img_obj, kernel_size=5):

	"""
	Description:
	------------
		Blur filter.
		Blurs the edges and grains in the image, can be used for reducing noise.

	Params:
	-------
	img_path : str
		path to the image to be processed
	kernel_size : int (Default : 5)
		size of the kernel to use for applying filter on the image.


	Returns:
	--------
	Processed image
	"""

	#Blur filter, AKA box filter

	#Normalization
	img = img_obj
	kernel = (kernel_size, kernel_size)
	filteredImg = cv2.blur(img, kernel)

	return filteredImg

#--------------------------------------------




def laplacian_filter_img(img_obj, kernel_size=7):

	"""
	Description:
	------------
		Edge detection filter, laplacian filter used after Guassian blur

	Params:
	-------
	img_path : str
		path to the image to be processed
	kernel_size : int (Default : 7)
		size of the kernel to use for applying filter on the image.


	Returns:
	--------
	Processed image
	"""

	#Edge detection filter, laplacian filter used after Guassian blur

	img = img_obj
	kernel = (kernel_size, kernel_size)
	blur = cv2.GaussianBlur(img,kernel,0)
	filteredImg = cv2.Laplacian(blur,cv2.CV_64F, ksize=kernel_size)
	filteredImg = ((filteredImg/filteredImg.max())*255).astype(np.uint8)

	return filteredImg


def canny_edge_filter_img(img_obj, minVal=100, maxVal=200):

	"""
	Description:
	------------
		Edge detection filter, Canny Edge filter is used.

	Params:
	-------
	img_path : str
		path to the image to be processed
	minVal : int (Default : 100)
		Minimum threshold value.
	maxVal : int (Default : 200)
		Maximum threshold value.

	Returns:
	--------
	Processed image
	"""

	filteredImg = cv2.Canny(img_obj, minVal, maxVal)

	return filteredImg


def filtering_app():

	st.header("Filtering Techniques")
	st.markdown("> An image filter is a technique through which size, colors, \
		shading and other characteristics of an image are altered. \
		An image filter is used to transform the image using different graphical editing techniques.")
	st.markdown("---")

	inp_img = st.file_uploader("Upload your image, (jpg, png)", type=['jpg','png', 'jpeg'])

	if inp_img:
		img_pil = Image.open(inp_img)
		img_cv2 = np.array(img_pil)
		
		st.image(img_pil, caption = "Original Image", use_column_width=True)
		st.markdown("---")

		filter_type = st.selectbox("Select Filter Type",
			("Blur_Filter", "Canny_Edge_Filter", "Laplacian_Filter"))

		if filter_type == "Blur_Filter":
			kernel_size = st.slider("kernel Size", 1, 11, 5, 1)
			out_img = blur_filter_img(img_cv2, kernel_size)
			out_img_pil = Image.fromarray(out_img)
			
			st.subheader("Filtered Image : Blur/Box Filter")
			st.image(out_img_pil, caption = "Image after Filtering", use_column_width=True)

		elif filter_type == "Laplacian_Filter":
			st.info("Kernel value should be odd")
			kernel_size = st.slider("kernel Size", 1, 11, 7, 1)
			out_img = edge_filter_img(img_cv2, kernel_size)
			out_img_pil = Image.fromarray(out_img)
			
			st.subheader("Filtered Image : Laplacian Filter")
			st.image(out_img_pil, caption = "Image after Filtering", use_column_width=True)

		elif filter_type == "Canny_Edge_Filter":
			minVal = st.slider("Minimum Thresold Value", 1, 300, 100, 1)
			maxVal = st.slider("Maximum Thresold Value", 1, 300, 200, 1)
			out_img = canny_edge_filter_img(img_cv2, minVal, maxVal)
			out_img_pil = Image.fromarray(out_img)
			
			st.subheader("Filtered Image : Canny Edge Filter")
			st.image(out_img_pil, caption = "Image after Filtering", use_column_width=True)