'''
@Author : TeJas.Lotankar

This contains main functions for calling untils and model selections for execution
'''




# imports 

from Custom_clf.imgClf_utils import get_train_img, preprocess_img, predict_on_model
from Custom_clf.imgClf_modelSelection import load_model
from keras.callbacks import ReduceLROnPlateau
import keras
import streamlit as st
import pandas as pd



class MyCallback(keras.callbacks.Callback):
	def __init__(self, x_test, num_epochs):
		self._num_epochs = num_epochs
		self._sample_tests = x_test[0:10]
	def on_train_begin(self, logs=None):
		st.header('Progress')
		# self._summary_chart = self._create_chart('area', 300)
		st.header('Percentage Complete')
		self._progress = st.empty()
		self._progress.progress(0)
		st.header('Current Epoch')
		self._epoch_header = st.empty()
		# st.header('A Few Tests')
		# self._sample_test_results = st.empty()
		# self._sample_test_results.dataframe(self._sample_tests)
	def on_epoch_begin(self, epoch, logs=None):
		self._epoch = epoch
		self._epoch_header.text(f'Epoch in progress: {epoch}')
	def on_batch_end(self, batch, logs=None):
		rows = pd.DataFrame([[logs['mean_squared_error']]],
			columns=['mean_squared_error'])
		if batch % 100 == 99:
			self._summary_chart.add_rows(rows)
		batch_percent = logs['batch'] * logs['size'] / self.params['samples']
		percent = self._epoch / self._num_epochs + (batch_percent / self._num_epochs)
		self._progress.progress(math.ceil(percent * 100))
	def on_epoch_end(self, epoch, logs=None):
		t = self._sample_tests
		prediction = np.round(self.model.predict([t.user_id, t.item_id]),0)
		self._sample_tests[f'epoch {epoch}'] = prediction
		self._sample_test_results.dataframe(self._sample_tests)
	
		return st.DeltaConnection.get_connection().get_delta_generator()._native_chart(epoch_chart)


def show_graphs(model_hist):
	'''
	//TODO
	'''

	st.subheader("Training Summary")
	# summarize history for accuracy
	plt.plot(model_hist.history['acc'])
	plt.plot(model_hist.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(model_hist.history['loss'])
	plt.plot(model_hist.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	st.pyplot()


# @st.cache
def train_model(train_img_path, model="inceptionnet", img_size=(299, 299, 3), num_epoch=10, batch_size=8, loss_function='categorical_crossentropy', optimizer='Adam'):
	'''
	Read the images from diretory
	Process the images
	Loads the model
	Train the model

	Parameters
	-------------
	model : Keras model
		Model name to be used for training and classification
		(Default "inceptionResNetV2")

	train_img_path : str
		Path to the directory containing images for training.
		Directory should contain folders named by labels and images of the same.

	img_size : tuple (299, 299, 3)
		Size of image to be processed into while training for classification.
		(Default set for InceptionResNetV2 i.e. (299, 299, 3))

	num_epoch : int
		Number of epochs to be performed while training model
		(Default set to 10)

	batch_size : int
		Number of images to be processed at a time. (Default to 8)

	loss_function : keras loss function
		//TODO

	optimizer : keras optimizer
		//TODO


	Returns
	---------
	- Trained keras model with weights.
	- History of trained model.
	- Evaluation of model on test data.
	- Class indices

	'''

	# Reading images
	img_df = get_train_img(train_img_path)

	test_df = img_df[0:20]

	# Processing images
	train_data, val_data, test_data, class_indices = preprocess_img(img_df=img_df, img_size=(img_size[0], img_size[1]), batch_size=batch_size)

	# loading model
	model_train = load_model(class_num=len(class_indices), img_size=img_size, loss_function=loss_function, optimizer=optimizer)

	
	learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
											patience=2,
											verbose=1,
											factor=0.5,
											min_lr=0.00001)

	#Setting callback for progress bar
	progress_callback = MyCallback(test_df, num_epoch)

	# Training model
	model_history = model_train.fit_generator(generator=train_data,
								validation_data=val_data,
								validation_steps=4,
								epochs=num_epoch,
								verbose=1,
								steps_per_epoch=train_data.samples//batch_size,
								use_multiprocessing=True,
								workers=-1,
								callbacks=[learning_rate_reduction, progress_callback])

	# Testing model on test data
	model_eval = model_train.evaluate_generator(test_data, steps=test_data.samples//batch_size, verbose=1)

	# Showing training graphs
	show_graphs(model_history)


	return model_train, model_history, model_eval, class_indices
	
