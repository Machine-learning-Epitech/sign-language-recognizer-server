from flask import Flask, jsonify, make_response, request
from numpy import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras import layers
import tensorflow as tf
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from flask_cors import CORS


# from flask.ext.bcrypt import Bcrypt
import os

classes = {
    "0": ["A", "a"],
    "1": ["B", "b"],
    "2": ["C", "c"],
    "3": ["D", "d"],
    "4": ["DEL", "del"],
    "5": ["E", "e"],
    "6": ["F", "f"],
    "7": ["G", "g"],
    "7": ["H", "h"],
    "8": ["I", "i"],
    "9": ["J", "j"],
    "10": ["K", "k"],
    "11": ["L", "l"],
    "12": ["M", "m"],
    "13": ["N", "n"],
    "14": ["Nothing", "nothing"],
    "15": ["O", "o"],
    "16": ["P", "p"],
    "17": ["Q", "q"],
    "18": ["R", "r"],
    "19": ["S", "s"],
    "20": ["Space", "space"],
    "21": ["T", "t"],
    "22": ["U", "u"],
    "23": ["V", "v"],
    "24": ["W", "w"],
    "25": ["X", "x"],
    "26": ["Y", "y"],
    "27": ["Z", "z"],
    "28": ["_", " "],
    "29": [".", ","],
}

model = load_model("my_model")

def decode_predictions(preds, top=28):
    if len(preds.shape) != 2 or preds.shape[1] != 30: # your classes number
        raise ValueError('`decode_predictions` expects '
                        'a batch of predictions '
                        '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(classes[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def predict(image_path, IMG_SIZE=200):
    image = Image.open(image_path)
    # resize image to target size
    image = image.resize((IMG_SIZE, IMG_SIZE))
    # image = normalization_layer()
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = tf.expand_dims(image, axis =-1)
    # Predict image using model
    resp = model.predict(image)
    return decode_predictions(resp, top=28)[0]

app = Flask(__name__)
cors = CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})
@app.route("/predict", methods=["POST"])
def predictRoute():
    if request.files:
        file = request.files["file"]
        filename = secure_filename(file.filename)   
        file.save(filename)
        prediction = predict(filename)
        os.remove(filename)
        return jsonify({"result": prediction[0][0]})
    return;


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv('PORT', 5002) ,debug=True)