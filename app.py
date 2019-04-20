#!flask/bin/python

import os
import cv2 as cv
import numpy as np
import io
import tensorflow as tf
import keras
from flask import Flask, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.applications import ResNet50
from keras.models import model_from_json
from keras.utils import CustomObjectScope

def create_app():
  # custom metrics
  def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

  def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

  def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

  # load our model 
  def load_model():
    global model, graph
    with CustomObjectScope({'recall': recall, 'precision':precision,'f1':f1}): 
      model = keras.models.load_model('modelV2.h5')
    graph = tf.get_default_graph()


  def prepare_image(image, target):
    # convert numpy array to image
    image = cv.imdecode(image, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image,target)
    image = image/255.0
    image = image.reshape(1,150,150,-1)
    return image
  
  load_model()

  app = Flask(__name__)

  @app.route("/predict", methods=['GET', "POST"])
  def predict():
    # initialize the data dictionary that will be returned
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
      if request.files.get("image"):
        # read image file string data
        file = request.files["image"].read()

        #convert string data to numpy array
        np_image = np.fromstring(file, np.uint8)

        # preprocess the image and prepare it for classification
        image = prepare_image(np_image, target=(150, 150))
        
        # classify the input image   
        with graph.as_default():
          preds = model.predict(image, verbose=1)
          data["predictions"] = str(preds[0][0])

        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return jsonify(data)
  return app

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
  print(("* Loading Keras model and Flask starting server..."
      "please wait until server has fully started"))
  load_model() 
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port)



