"""
@Author : TeJas.Lotankar
"""

import streamlit as st


def get_info_actionRecognition():
	
	st.header("Description:")
	st.info("Action recognition task involves the identification of different actions \
		from video clips (a sequence of 2D frames) where the action may or may not be \
		performed throughout the entire duration of the video. This seems like a \
		natural extension of image classification tasks to multiple frames and then \
		aggregating the predictions from each frame.")