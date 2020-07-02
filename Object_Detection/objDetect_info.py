"""
@Author : TeJas.Lotankar
"""

import streamlit as st


def get_info_objDetect():
	
	st.header("Description:")
	st.info("Object detection is a computer vision technique for locating instances of \
		objects in images or videos. Object detection algorithms typically leverage \
		machine learning or deep learning to produce meaningful results. \
		When humans look at images or video, we can recognize and locate objects of \
		interest within a matter of moments. The goal of object detection is to replicate \
		this intelligence using a computer.")

	st.markdown("---")
	st.header("Architectures:")
	st.subheader("VOC Dataset:")
	st.info("mAP : Mean Average Precision, Range from 0-100, higher the better.")
	st.info("Each bounding box will have likelyhood score of box containing object.\
		Based on predictions, Precision-Recall(PR) curve is calculated and Averaege Precision(AP)\
		is area under PR curve.\
		First AP is calculated for each class and averaged over different classes, which results\
		in mAP.")
	vocData = '''
| Model                    | mAP  |
|--------------------------|------|
| ssd_512_resnet50_v1_voc  | 80.1 |
| ssd_512_mobilenet1.0_voc | 75.4 |
| yolo3_darknet53_voc      | 81.5 |
| yolo3_mobilenet1.0_voc   | 75.8 |
	'''
	st.markdown(vocData)


	st.subheader("COCO Dataset:")
	st.info("The COCO metric, Average Precision (AP) with IoU threshold 0.5:0.95 (averaged 10 values, \
		AP 0.5:0.95), 0.5 (AP 0.5) and 0.75 (AP 0.75) are reported together in the format\
		 (AP 0.5:0.95)/(AP 0.5)/(AP 0.75).")

	cocoData = '''
| Model                     | Box AP         |
|---------------------------|----------------|
| ssd_512_resnet50_v1_coco  | 30.6/50.0/32.2 |
| ssd_512_mobilenet1.0_coco | 21.7/39.2/21.3 |
| yolo3_darknet53_coco      | 36.0/57.2/38.7 |
| yolo3_mobilenet1.0_coco   | 28.6/48.9/29.9 |
	'''
	st.markdown(cocoData)