"""
@Author : TeJas.Lotankar
"""

# imports
from gluoncv import model_zoo, data, utils
import mxnet as mx

from matplotlib import pyplot as plt
from PIL import Image
import cv2


def get_img_segment(img_object, model="mask_rcnn_resnet50_v1b_coco", thresh=0.7, is_limited_classes=False, class_list=[]):

	net = model_zoo.get_model(model, pretrained=True)

	if is_limited_classes:
		net.reset_class(class_list, reuse_weights=class_list)

	try:
		img_object = utils.download(img_object)
	except ValueError:
		pass

	x, orig_img = data.transforms.presets.rcnn.load_test(img_object)

	ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

	# paint segmentation mask on images directly
	width, height = orig_img.shape[1], orig_img.shape[0]
	masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores, thresh=thresh)
	orig_img = utils.viz.plot_mask(orig_img, masks)

	# identical to Faster RCNN object detection
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids, thresh=thresh,
	                         class_names=net.classes, ax=ax)
	
	return ax


