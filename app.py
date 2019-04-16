#!flask/bin/python

import os
import cv2 as cv
from flask import Flask, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.applications import ResNet50
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import keras
from keras import backend as K


#metrics

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
  K.clear_session()
  tf.reset_default_graph()

  global model, graph
#  model = ResNet50(weights="imagenet")

  model = tf.keras.models.load_model('cancer_classifier.h5', custom_objects={'recall': recall, 'precision':precision,'f1':f1}, compile=True)
  graph = tf.get_default_graph()

#  return model, graph



def prepare_image(image, target):
  # convert numpy array to image
  image = cv.imdecode(image, cv.IMREAD_GRAYSCALE)
#  image = cv.imread(image, cv.IMREAD_GRAYSCALE)
  image = cv.resize(image,target)
  image = image/255.0
  image = image.reshape(1,150,150,-1)
  
  return image
  
#    # if the image mode is not RGB, convert it
#    if image.mode != "RGB":
#        image = image.convert("RGB")
#
#    # resize the input image and preprocess it
#    image = image.resize(target)
#    image = img_to_array(image)
#    image = np.expand_dims(image, axis=0)
#    image = imagenet_utils.preprocess_input(image)
#
#    # return the processed image
#    return image

UPLOAD_FOLDER = './image'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'svg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#@app.route('/')
#def index():
#  return "hello world"


@app.route("/predict", methods=['GET', "POST"])
def predict():
  # initialize the data dictionary that will be returned from the
  # view
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

      
      
#      image = request.files["image"]
#
#      image.save(secure_filename(image.filename))
#      image = prepare_image(image.filename, target=(150, 150))


      
#      image = request.files["image"].read()
#      image = Image.open(io.BytesIO(image))


      # classify the input image and then initialize the list
      # of predictions to return to the client
      
#      with graph.as_default():
      
      with graph.as_default():
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
#          is_variable_initialized
#          model._make_predict_function()
          preds = model.predict(image, verbose=1)

#      model, graph = load_model()    
          data["predictions"] = str(preds[0][0])

      # loop over the results and add them to the list of
      # returned predictions
#      results = imagenet_utils.decode_predictions(preds)
#      for (imagenetID, label, prob) in results[0]:
#          r = {"label": label, "probability": float(prob)}
#          data["predictions"].append(r)

      # indicate that the request was a success
      data["success"] = True

  # return the data dictionary as a JSON response
  return jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model() 
    app.run()



#def allowed_file(filename):
#    return '.' in filename and \
#           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#@app.route('/', methods=['GET', 'POST'])
#def upload_file():
#    if request.method == 'POST':
#      # check if the post request has the file part
#      if 'file' not in request.files:
#          flash('No file part')
#          return redirect(request.url)
#      file = request.files['file']
#      # if user does not select file, browser also
#      # submit a empty part without filename
#      if file.filename == '':
#          flash('No selected file')
#          return redirect(request.url)
#      if file and allowed_file(file.filename):
#          filename = secure_filename(file.filename)
#          file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#
#          return redirect(url_for('uploaded_file',
#                                  filename=filename))
#
#    return '''
#    <!doctype html>
#    <title>Upload new File</title>
#    <h1>Upload new File</h1>
#    <form method=post enctype=multipart/form-data>
#      <p><input type=file name=file>
#         <input type=submit value=Upload>
#    </form>
#    '''
#  
#@app.route('/uploads/<filename>')
#def uploaded_file(filename):
#    return send_from_directory(app.config['UPLOAD_FOLDER'],
#                               filename)
#  
  

#tasks = [
#    {
#        'id': 1,
#        'title': u'Buy groceries',
#        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol', 
#        'done': False
#    },
#    {
#        'id': 2,
#        'title': u'Learn Python',
#        'description': u'Need to find a good Python tutorial on the web', 
#        'done': False
#    }
#]
#
#@app.route('/todo/api/v1.0/tasks', methods=['GET'])
#def get_tasks():
#    return jsonify({'tasks': tasks})
#  
#@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
#def get_task(task_id):
#    task = [task for task in tasks if task['id'] == task_id]
#    if len(task) == 0:
#        abort(404)
#    return jsonify({'task': task[0]})
#  
#@app.errorhandler(404)
#def not_found(error):
#    return make_response(jsonify({'error': 'Not found'}), 404)
#
#@app.route('/todo/api/v1.0/tasks', methods=['POST'])
#def create_task():
#    if not request.json or not 'title' in request.json:
#        abort(400)
#    task = {
#        'id': tasks[-1]['id'] + 1,
#        'title': request.json['title'],
#        'description': request.json.get('description', ""),
#        'done': False
#    }
#    tasks.append(task)
#    return jsonify({'task': task}), 201

#if __name__ == '__main__':
#  app.run(debug=True)
