"""
@Author : TeJas.Lotankar

Description:
------------
	Utils and helper functions for Object Detection.
"""


# imports
from gluoncv import model_zoo, data, utils
import mxnet as mx

from matplotlib import pyplot as plt
from PIL import Image
import cv2



def get_object_detection(img_object, model="yolo3_darknet53_coco", thresh=0.7, is_limited_classes=False, class_list=[]):
	'''
	//TODO

	'''

	net = model_zoo.get_model(model, pretrained=True)

	if is_limited_classes:
		net.reset_class(class_list, reuse_weights=class_list)

	try:
		img_object = utils.download(img_object)
	except ValueError:
		pass

	if "yolo" in model:
		x, img = data.transforms.presets.yolo.load_test(img_object, short=512)
	elif "ssd" in model:
		x, img = data.transforms.presets.ssd.load_test(img_object, short=512)
	

	class_IDs, scores, bounding_boxs = net(x)

	ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
	                         class_IDs[0], class_names=net.classes, thresh=thresh)
	return ax