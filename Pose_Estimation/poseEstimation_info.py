"""
@Author : TeJas.Lotankar
"""

import streamlit as st


def get_info_poseEstimation():
	
	st.header("Description:")
	st.info("Human Pose Estimation is defined as the problem of localization of \
		human joints (also known as keypoints - elbows, wrists, etc) in images or videos. \
		It is also defined as the search for a specific pose in space of all articulated poses.")

# --------------------------------------------------------------------------------
	st.markdown("---")
	st.header("Object Detection Networks:")
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

# -----------------------------------------------------------------------------------
	st.markdown("---")
	st.header("Pose Estimation Networks:")
	st.markdown("> [Object Keypoint Similarity AP](http://cocodataset.org/#keypoints-eval)")

	poseData = '''
| Model                         | OKS AP         | OKS AP (With flip) |
|-------------------------------|----------------|--------------------|
| simple_pose_resnet18_v1b      | 66.3/89.2/73.4 | 68.4/90.3/75.7     |
| simple_pose_resnet50_v1b      | 71.0/91.2/78.6 | 72.2/92.2/79.9     |
| simple_pose_resnet101_v1b     | 72.4/92.2/79.8 | 73.7/92.3/81.1     |
| simple_pose_resnet152_v1b     | 72.4/92.1/79.6 | 74.2/92.3/82.1     |
| alpha_pose_resnet101_v1b_coco | 74.2/91.6/80.7 | 76.7/92.6/82.9     |
	'''
	st.markdown(poseData)