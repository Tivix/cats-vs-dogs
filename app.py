import os
import json
import cv2
import requests

from flask import Flask, request
from flask_cors import CORS

import keras
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model, model_from_json
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image
import numpy as np

import tensorflow as tf


app = Flask(__name__)

# Allow 
CORS(app)

# Allowed file extransions
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

@app.route("/")
def hello():
	return "Hello World!"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':

		# Check if the post request has the file part
		if 'file' not in request.files:
			return "No file part"
		file = request.files['file']

		if file and allowed_file(file.filename):

			# Get image from frontend stream
			image_from_frontend = Image.open(request.files['file'].stream)
			
			# Send uploaded image for prediction
			predicted_image_class = predict_img(image_from_frontend)

	# Send predicted class back to frontend
	return json.dumps(predicted_image_class)

def predict_img(image_from_frontend):
	K.clear_session()

	# Choose same image size as the model is trained on
	image_size = (150, 150)

	json_file = open('model_new.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# Add your available classes for predicition
	classes = {0:'cats', 1:'dogs'}

	model = VGG16(include_top=False, weights='imagenet')

	# Transform image from bytes to numpy array
	img = np.array(image_from_frontend)
	img = cv2.resize(img, dsize=image_size)
	img = img / 255
	img = np.expand_dims(img, axis=0)

	bottleneck_prediction = model.predict(img)

	# Load weights into a new model
	loaded_model.load_weights("cats_dogs_model.h5")

	# Compile the loaded model
	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	
		# use the bottleneck prediction on the top model to get the final classification  
	class_predicted = loaded_model.predict_classes(bottleneck_prediction)  

	# Get predicted class
	prediction_label = classes[class_predicted[0][0]]

	return prediction_label

if __name__ == "__main__":
	app.run(debug=True)
