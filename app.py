from flask import Flask, render_template, request, jsonify,send_file, Response
from PIL import Image
import cv2
import numpy as np
from flask_cors import CORS
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# API KEYS for APIs
API_KEY = "oziapps_41d8d9d2bb4c641b65d3515cbc27f99f"

""" Global parameters """
H = 512
W = 512
host = os.environ.get("FLASK_HOST", "172.16.0.94")
port = int(os.environ.get("FLASK_PORT", 9000))

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


app = Flask(__name__)
CORS(app)


""" Loading model: DeepLabV3+ """
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = tf.keras.models.load_model("model.h5")


@app.route("/")
def index():
    return render_template("index.html")        

# Background removal API
@app.route("/removebg", methods=["POST"])
def remove_bg():
    print('Removing Bg')
    print(request.headers.get('Authorization'))
    if request.headers.get('Authorization') == f'Bearer {API_KEY}':
        print("Removing the background")    
        # Get the uploaded image from the request
        image = request.files["image"]
        # Read image using Pillow
        pil_image = Image.open(image)

        # Get the current datetime
        current_datetime = datetime.now()

        # Convert the datetime to a string using a specific format
        formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
        image_name = image.filename.replace(".jpg", "") + '_'+ formatted_datetime + '.jpg';
        
    
        """ Seeding """
        np.random.seed(42)
        tf.random.set_seed(42)
    
        """ Directory for storing files """
        create_dir("remove_bg")
    
        """ Extracting name """
        name = image_name.split(".")[0]
    
        # Convert Pillow image to NumPy array (OpenCV format)
        image = np.array(pil_image)  #CV image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
    
        """ Prediction """
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)
        y = y > 0.5
    
        photo_mask = y
        background_mask = np.abs(1-y)
    
        masked_photo = image * photo_mask
        background_mask = np.concatenate([background_mask, background_mask, background_mask], axis=-1)
        background_mask = background_mask * [0, 0, 0]
        final_photo = masked_photo + background_mask
        temp = 'remove_bg/'+ name + '.jpg'
        print(temp)
        
        cv2.imwrite(temp, final_photo)
    
        processed_image_filename = image_name  # Replace with the actual processed filename.
        processed_image_path = os.path.join("remove_bg", processed_image_filename)
    
        # Replace backslashes with forward slashes in the path for the URL.
        processed_image_url = processed_image_path.replace("\\", "/")
        processed_image_url = f"http://{host}:{port}/{processed_image_url}"
    
        # Return the URL or path of the processed image as a JSON response.
        response = {"processed_image_url": processed_image_url}
        return jsonify(response)
    else:
        return jsonify(message='Unauthorized'), 401

@app.route("/remove_bg/<filename>")
def serve_image(filename):
    # Replace 'path/to/' with the correct path to your image folder.
    image_path = os.path.join("remove_bg", filename)
    print(image_path)
    
    if not os.path.isfile(image_path):
        # Return an error image or a default image if the requested image doesn't exist.
        default_image_path = "images/Not_found.jpg"  # Replace with the path to your default image.
        image_path = default_image_path

    # Determine the image mimetype based on the file extension.
    # You may need to handle additional image formats as needed.
    mimetype = "image/jpeg" if filename.lower().endswith(".jpg") else "image/png"

    return send_file(image_path, mimetype=mimetype)

if __name__ == "__main__":
    
    app.run(host=host, port=port, debug=True)



