"""
@Author : TeJas.Lotankar

Description:
------------
	Utils and helper functions for Action Recognition.
"""


# imports 
import mxnet as mx
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model

from gluoncv.utils.filesystem import try_import_decord

def get_action_recognition(video_obj, model_arch = "slowfast_4x16_resnet50_kinetics400"):
	'''
	//TODO
	'''
	# starting decord
	decord = try_import_decord()

	net = get_model(model_arch, pretrained=True)

	try:
		video_obj = utils.download(video_obj)
	except ValueError:
		pass

	vr = decord.VideoReader(video_obj)

	if "slowfast" in model_arch:
		fast_frame_id_list = range(0, 64, 2)
		slow_frame_id_list = range(0, 64, 16)
		frame_id_list = list(fast_frame_id_list) + list(slow_frame_id_list)
	else:
		frame_id_list = range(0, 64, 2)

	print("=========Reached here============")

	video_data = vr.get_batch(frame_id_list).asnumpy()
	clip_input = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]

	if "inceptionv3" in model_arch:
		transform_fn = video.VideoGroupValTransform(size=299, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		clip_input = transform_fn(clip_input)
		clip_input = np.stack(clip_input, axis=0)
		if "slowfast" in model_arch:
			clip_input = clip_input.reshape((-1,) + (36, 3, 340, 450))
		else:
			clip_input = clip_input.reshape((-1,) + (32, 3, 340, 450))
		clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
	else:
		transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		clip_input = transform_fn(clip_input)
		clip_input = np.stack(clip_input, axis=0)
		if "slowfast" in model_arch:
			clip_input = clip_input.reshape((-1,) + (36, 3, 224, 224))
		else:
			clip_input = clip_input.reshape((-1,) + (32, 3, 224, 224))
		clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

	pred = net(nd.array(clip_input))

	classes = net.classes
	topK = 5
	ind = nd.topk(pred, k=topK)[0].astype('int')
	resList = []
	

	for i in range(topK):
		resList.append( [classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()] )

	resDF = pd.DataFrame(resList, columns=["class", "prob"])
	return resDF