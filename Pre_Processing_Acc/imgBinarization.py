"""
@Author : TeJas.Lotankar

Description:
------------
	Functions for binarizing images.

"""


import cv2
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
import streamlit as st



def get_otsu(img,min_thresh=127,max_thresh=255):
	ret, imgf = cv2.threshold(img, min_thresh, max_thresh, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return imgf

def get_adaptive(img,thresh=255):
	ada_th = cv2.adaptiveThreshold(img,thresh,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	return ada_th

def img_bin_app():
	st.header("Image Binarization")

	st.markdown("> Image binarization is the process of taking a grayscale image and converting it \
		to black-and-white, essentially reducing the information contained within \
		the image from 256 shades of gray to 2: black and white, a binary image.")

	st.markdown("---")
	
	inp_img = st.file_uploader("Upload your image, (jpg, png)", type=['jpg','png', 'jpeg'], key="binarization")

	if inp_img:
		img_pil = Image.open(inp_img)
		img_cv2 = np.array(img_pil)
		gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
		st.image(img_pil, caption = "Original Image", use_column_width=True)
		st.markdown("---")

		bin_type = st.selectbox("Select Binarization Type",
			("Otsu's Binarization", "Adaptive Threshold"))

		to_blur = st.checkbox("Apply Guassian Blur")
		if to_blur:
			gray_image = cv2.GaussianBlur(gray_image,(5,5),0)

		if bin_type == "Otsu's Binarization":
			min_thresh = st.slider("Minimum Threshold", 0, 255, 127, 1)
			max_thresh = st.slider("Maximum Threshold", 0, 255, 255, 1)

			out_img = get_otsu(gray_image,min_thresh,max_thresh)
			out_img_pil = Image.fromarray(out_img)
			
			st.subheader("Binarized Image : Otsu's Binarization")
			st.image(out_img_pil, caption = "Image after binarization", use_column_width=True)
		else:
			thresh = st.slider("Threshold", 0, 255, 255, 1)

			out_img = get_adaptive(gray_image,thresh)
			out_img_pil = Image.fromarray(out_img)
			
			st.subheader("Binarized Image : Adaptive Threshold")
			st.image(out_img_pil, caption = "Image after binarization", use_column_width=True)

# img_bin_app()