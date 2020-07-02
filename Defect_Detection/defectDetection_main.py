'''
@Author : TeJas.Lotankar

Displaying defect detection with streamlit
'''


import streamlit as st
import numpy as np
from PIL import Image
import random
import time


DATA_PATH = "Defect_Detection/imgData/"
DATA_MAPPING = [
	{'og': 'small_nut_defect_og.JPG', 'res': 'small_nut_defect.jpg'}, 
	{'og': 'Small_Big_Nut_og.JPG', 'res': 'Small_Big_Nut.jpg'}, 
	{'og': '2_cross_bolts_og.JPG', 'res': '2_cross_bolts.jpg'}, 
	{'og': '2_big_nut_defect_og.JPG', 'res': '2_big_nut_defect.jpg'}, 
	{'og': '5_small_nuts_og.JPG', 'res': '5_small_nuts.jpg'}, 
	{'og': 'small_nut_false_defect_og.JPG', 'res': 'small_nut_false_defect.jpg'}, 
	{'og': 'big_nut_defect_some_false_og.JPG', 'res': 'big_nut_defect_some_false.jpg'}, 
	{'og': '2_bolts_og.JPG', 'res': '2_bolts.jpg'}, 
	{'og': 'Nut_Small_001_og.JPG', 'res': 'Nut_Small_001.jpg'}, 
	{'og': 'nut_bolt_big_small_defect_og.JPG', 'res': 'nut_bolt_big_small_defect.jpg'}, 
	{'og': 'Small_small_nut_og.JPG', 'res': 'Small_small_nut.jpg'}]

VIDEO_RESULT = DATA_PATH + "/NutBolt_result_2000_001.mp4"

def defectDetection_main_app():


	st.title("Defect Detection")
	st.info("Defect detection is use case built using object detection.\
		Here defects in the products are detected real time.")
	st.markdown("---")
	st.header("Results:")

	res_choice = st.radio("Select results you want:",
		("Image_Data", "Video_Data"))

	if res_choice == "Image_Data":
		if st.button("Get result"):
			temp_img = random.choice(DATA_MAPPING)
			og_img = DATA_PATH+temp_img.get("og")
			res_img = DATA_PATH+temp_img.get("res")
			st.image(Image.open(og_img), caption="Original Image", use_column_width=True)
			
			with st.spinner("Processing image.."):
				prog = st.progress(0)
				for pr in range(100):
					time.sleep(random.uniform(0.1,0.4))
					prog.progress(pr+1)
			st.success("Result:")
			st.image(Image.open(res_img), caption="Results", use_column_width=True)

	elif res_choice == "Video_Data":
		with open(VIDEO_RESULT, 'rb') as v_file:
			vid_bytes = v_file.read()

		st.video(vid_bytes)



