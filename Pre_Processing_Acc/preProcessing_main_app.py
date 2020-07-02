'''
@Author : TeJas.Lotankar

Description:
------------
	Main app for calling all functions in pre-preocessing acc.
'''

# imports 
import streamlit as st


from Pre_Processing_Acc.Gamma_Correction import adjust_gamma_app
from Pre_Processing_Acc.imgBinarization import img_bin_app
from Pre_Processing_Acc.Histogram_Equalization import equalize_hist_app
from Pre_Processing_Acc.imgDenoise import denoising_app
from Pre_Processing_Acc.imgNormalize import norm_img_app
from Pre_Processing_Acc.imgFilters import filtering_app
from Pre_Processing_Acc.preProcessing_info import get_info_preproc




def pp_main_app():
	st.title("Image Pre-processing Accelerator")
	st.markdown("---")


	features = {
				"About" : "get_info_preproc()",
				"Gamma_Correction" : "adjust_gamma_app()",
				"Binarization" : "img_bin_app()",
				"Histogram_Equalization" : "equalize_hist_app()",
				"Image_De-Nosing" : "denoising_app()",
				"Image_Normalization" : "norm_img_app()",
				"Image_Filtering" : "filtering_app()"
	}

	st.sidebar.header("Pre-processing Accelerator")
	sel_func = st.sidebar.selectbox("Select fucntion",
		("About", "Gamma_Correction", "Binarization", "Image_Filtering", 
			"Histogram_Equalization", "Image_De-Nosing", "Image_Normalization"))

	exec(features[sel_func])



	st.sidebar.markdown("---")

# pp_main_app()