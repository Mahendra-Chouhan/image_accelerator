"""
@Author : TeJas.Lotankar

Description:
------------
	Utils and helper functions for Pose Estimation.
"""



# imports 
import mxnet as mx
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
from PIL import Image
import cv2


def get_pose_estimation(img_object, detector_model="yolo3_mobilenet1.0_coco", pose_model="simple_pose_resnet18_v1b" , box_thresh=0.5, keypoint_thresh=0.2):
	'''
	//TODO
	'''
	detector = model_zoo.get_model(detector_model, pretrained=True)

	pose_net = model_zoo.get_model(pose_model, pretrained=True)

	# Loading weights for only person class
	detector.reset_class(["person"], reuse_weights=['person'])

	try:
		img_object = utils.download(img_object)
	except ValueError:
		pass

	if "yolo" in detector_model:
		x, img = data.transforms.presets.yolo.load_test(img_object, short=512)
	elif "ssd" in detector_model:
		x, img = data.transforms.presets.ssd.load_test(img_object, short=512)

	class_IDs, scores, bounding_boxs = detector(x)

	if "simple_pose" in pose_model:
		pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
		predicted_heatmap = pose_net(pose_input)
		pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
	elif "alpha_pose" in pose_model:
		pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs)
		predicted_heatmap = pose_net(pose_input)
		pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

	ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
							class_IDs, bounding_boxs, scores,
							box_thresh=box_thresh, keypoint_thresh=keypoint_thresh)

	return ax