# Gamma Correction

from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

# IMG_PATH = "./images/cat.jpg"

def adjust_gamma(img_path, gamma=1.6):
	# Lookup table is built to mapping the pixel values 
	# [0, 255] to adjust gamma values
	invGamma = 1.0/gamma
	lookupTable = np.array([( (i/255.0)**invGamma )*255 
		for i in np.arange(0, 256)]).astype("uint8")

	img = img_path

	# applying gamma correction using Lookup Table 
	outImg = cv2.LUT(img, lookupTable)

	return outImg



def adjust_gamma_app():
	st.header("Gamma Correction")

	st.markdown("> [Gamma correction](https://en.wikipedia.org/wiki/Gamma_correction), \
		or often simply gamma,\
		is a nonlinear operation used to encode and decode\
		 luminance or tristimulus values in video or still image systems")
	st.markdown("---")
	
	inp_img = st.file_uploader("Upload your image, (jpg, png)", type=['jpg','png', 'jpeg'])

	if inp_img:
		img_pil = Image.open(inp_img)
		img_cv2 = np.array(img_pil)
		st.image(img_pil, caption = "Original Image", use_column_width=True)

		st.markdown("---")
		gamma_val = st.slider("Select Gamma Value", -3., 3., 1.6, 0.1)
		
		out_img = adjust_gamma(img_cv2, gamma_val)
		out_img_pil = Image.fromarray(out_img)
		
		st.subheader("Gamma Changed")
		st.image(out_img_pil, caption = "Gamma Changed Image", use_column_width=True)
