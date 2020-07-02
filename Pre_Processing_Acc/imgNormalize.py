"""
@Author : TeJas.Lotankar
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

from PIL import Image
import streamlit as st

def norm_img(img, lower_val=130, higher_val=180):

    """
    Description:
    ------------
        For normalizing pixel values of the images.
        pixel values will be normalized using new lower(130) and higher(180) pixel values 
        instead of 0 and 255, and all values will be mapped between 130-180.

    Params:
    -------
    img_path : str
        path to the image to be processed
    lower_val : int (Default : 130)
        Lower pixels values.
    higher_val : int (Default : 180)
        Higher pixel values.


    Returns:
    --------
    Processed image
    """

    #Normalization
    normalizedImg = cv.normalize(img, None, lower_val, higher_val, cv.NORM_MINMAX)
    return normalizedImg


def norm_img_app():
    st.header("Image Normalization")
    st.markdown("> Image normalization is a typical process in image processing that changes\
     the range of pixel intensity values. Its normal purpose is to convert an input image\
      into a range of pixel values that are more familiar or normal to the senses, \
      hence the term normalization")
    st.markdown("---")
    
    inp_img = st.file_uploader("Upload your image, (jpg, png)", type=['jpg','png', 'jpeg'])

    if inp_img:
        img_pil = Image.open(inp_img)
        img_cv2 = np.array(img_pil)
        st.image(img_pil, caption = "Original Image", use_column_width=True)
        st.markdown("---")

        low_val = st.slider("Lower Value", 0, 255, 130, 1)
        high_val = st.slider("Search Window Size", 0, 255, 180, 1)

        out_img = norm_img(img_cv2,low_val, high_val)
        out_img_pil = Image.fromarray(out_img)
        
        st.subheader("Normalized Image")
        st.image(out_img_pil, caption = "Image after Normalization", use_column_width=True)

# norm_img_app()