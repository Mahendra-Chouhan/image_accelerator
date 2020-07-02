"""
@Author : TeJas.Lotankar

Description:
------------
	To equalize histogram of the image

Params:
-------
img_path : str
	path to the image to be processed


Returns:
--------
Image with equalized histogram

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
import streamlit as st

IMG_PATH = "./images/cat.jpg"

def equalize_hist(img_path, clip_limit=2.0):
	img = img_path
	#Creating object for CLAHE
	clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
	#Applying equalization
	outImg = clahe.apply(img)

	return outImg



def equalize_hist_app():
	st.header("Histogram equalization")

	st.markdown("> Histogram Equalization is a computer image processing technique used to improve contrast \
		in images. It accomplishes this by effectively spreading out the most frequent \
		intensity values, i.e. stretching out the intensity range of the image.")
	st.markdown("---")
	
	inp_img = st.file_uploader("Upload your image, (jpg, png)", type=['jpg','png', 'jpeg'])

	if inp_img:
		img_pil = Image.open(inp_img)
		img_cv2 = np.array(img_pil)
		gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
		st.image(img_pil, caption = "Original Image", use_column_width=True)
		st.markdown("---")

		clip_limit = st.slider("Select Clip Limit", 0., 10., 2.0, 0.1)
		
		out_img = equalize_hist(gray_image, clip_limit)
		out_img_pil = Image.fromarray(out_img)
		
		st.subheader("Histogram Changed")
		st.image(out_img_pil, caption = "Image after altered histogram", use_column_width=True)

# equalize_hist_app()