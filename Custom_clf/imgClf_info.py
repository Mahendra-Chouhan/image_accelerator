'''
@Author : TeJas.Lotankar
'''


import streamlit as st


def get_info_clf():
	st.title("Custom Image Classification")
	st.header("Description")
	st.info("This use case helps building custom image classfier using own images. \
		User can Provide own images for training, and get trained keras model for predictions.\
		Trained model can be used for testing on single images or label set of images.")
	st.header("How to use?")
	st.text("1. Create zip file of training image with following folder structure")

	st.text("train_files.zip \n\
	| \n\
	|___ cat (folder with same name of class) \n\
	|   |   image_001.jpg \n\
	|   |   image_002.jpg \n\
	|   |   image_003.jpg \n\
	|   | \n\
	|___ dog (folder with same name of class) \n\
	|   |   image_001.jpg \n\
	|   |   image_002.jpg \n\
	|   |   image_003.jpg \n\
	|   | \n\
	|___ another class (folder with same name of class) \n\
	|   |   image_001.jpg \n\
	|   |   image_002.jpg \n\
	|   |   image_003.jpg")

	st.text("\
		2. Upload zip file containing training images and wait for setup\n\
		3. Select Network architecture for training\n\
		4. Select number of epoch to train for\n\
		5. Select Batch Size for Training \n\
		6. Select Loss Function and Optimizer \n\
		7. Start training and wait for training to complete \n\
		8. After training is complete, start visualizing predicting with single image. \n")