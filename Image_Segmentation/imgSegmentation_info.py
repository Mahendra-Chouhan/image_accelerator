"""
@Author : TeJas.Lotankar
"""

import streamlit as st


def get_info_segment():
	
	st.header("Description:")
	st.info("Image segmentation is a commonly used technique in digital image processing and \
		analysis to partition an image into multiple parts or regions, often based on the \
		characteristics of the pixels in the image. Image segmentation could involve \
		separating foreground from background, or clustering regions of pixels based on \
		similarities in color or shape. For example, a common application of image segmentation \
		in medical imaging is to detect and label pixels in an image or voxels of a 3D volume \
		that represent a tumor in a patient’s brain or other organs.")

	st.markdown("---")
	st.header("Architectures:")
	st.info("mAP : Mean Average Precision, Range from 0-100, higher the better.")
	st.info("Each bounding box will have likelyhood score of box containing object.\
		Based on predictions, Precision-Recall(PR) curve is calculated and Averaege Precision(AP)\
		is area under PR curve.\
		First AP is calculated for each class and averaged over different classes, which results\
		in mAP.")
	st.info("The COCO metric, Average Precision (AP) with IoU threshold 0.5:0.95 (averaged 10 values, \
		AP 0.5:0.95), 0.5 (AP 0.5) and 0.75 (AP 0.75) are reported together in the format\
		 (AP 0.5:0.95)/(AP 0.5)/(AP 0.75).")

	archData = '''
| Model                        | Box AP         | Segment AP     |
|------------------------------|----------------|----------------|
| mask_rcnn_resnet50_v1b_coco  | 38.3/58.7/41.4 | 33.1/54.8/35.0 |
| mask_rcnn_resnet18_v1b_coco  | 31.2/51.1/33.1 | 28.4/48.1/29.8 |
| mask_rcnn_resnet101_v1d_coco | 41.3/61.7/44.4 | 35.2/57.8/36.9 |
	'''
	st.markdown(archData)