
"""
@Author : TeJas.Lotankar

Description:
------------
	To remove noise from the image.

Params:
-------
img_path : str
	path to the image to be processed


Returns:
--------
Processed image

"""

from PIL import Image
import streamlit as st

import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet
import skimage



def get_fastnNlMeanDenoising(img, templateWindowSize=7, searchWindowSize=21, h=4, hColor=3):
	outImg_cv2 = cv2.fastNlMeansDenoisingColored(img, None, 
										templateWindowSize, searchWindowSize,
										 h, hColor)
	return outImg_cv2

def get_tv_chambolle(img, weight=0.1):
	outImg_tv = denoise_tv_chambolle(img, weight=weight, multichannel=True)
	return outImg_tv

def get_wavelet(img, wavelet='db1', mode='soft'):
	outImg_wavelet = denoise_wavelet(img, wavelet=wavelet, mode=mode, multichannel=True)
	return outImg_wavelet


def denoising_app():
	st.header("Image De-Noising")
	st.markdown("> Image Denoising. One of the fundamental challenges in the field of image \
		processing and computer vision is image denoising, where the underlying goal is to \
		estimate the original image by suppressing noise from a noise-contaminated version of the image.")
	st.markdown("---")
	
	inp_img = st.file_uploader("Upload your image, (jpg, png)", type=['jpg','png', 'jpeg'])

	if inp_img:
		img_pil = Image.open(inp_img)
		img_cv2 = np.array(img_pil)
		st.image(img_pil, caption = "Original Image", use_column_width=True)
		st.markdown("---")

		denoise_type = st.selectbox("Select Denoising Type",
			("fastnNlMeanDenoising", "tv_chambolle", "wavelet"))

		if denoise_type == "fastnNlMeanDenoising":
			temp_window = st.slider("Template Window Size", 0, 20, 7, 1)
			search_window = st.slider("Search Window Size", 0, 30, 21, 1)
			h_val = st.slider("Regulator for strength for luminance", 0, 10, 3, 1)
			h_color_val = st.slider("Regulator for strength for luminance, color.", 0, 10, 3, 1)

			out_img = get_fastnNlMeanDenoising(img_cv2, temp_window,search_window, h_val, h_color_val)
			out_img_pil = Image.fromarray(out_img)
			
			st.subheader("Denoised Image : Fast and NL Mean Denoising")
			st.image(out_img_pil, caption = "Image after Denoising", use_column_width=True)

		elif denoise_type == "tv_chambolle":
			weight = st.slider("Denoising Weigth", 0.1, 10., 1., 0.1)
			
			out_img = get_tv_chambolle(img_cv2, weight)
			out_img_pil = Image.fromarray(skimage.util.img_as_ubyte(out_img))
			
			st.subheader("Denoised Image : TV Chambolle Denoising")
			st.image(out_img_pil, caption = "Image after Denoising", use_column_width=True)

		elif denoise_type == "wavelet":
			wavelet = st.selectbox("Select wavelet Type",
			('db1', 'db2', 'haar', 'sym9'))
			
			mode = st.selectbox("Type 0f denoising",
				('soft', 'hard'))


			out_img = get_wavelet(img_cv2, wavelet, mode)
			out_img_pil = Image.fromarray(skimage.util.img_as_ubyte(out_img))
			
			st.subheader("Denoised Image : Wavelet Denoising")
			st.image(out_img_pil, caption = "Image after Denoising", use_column_width=True)		
