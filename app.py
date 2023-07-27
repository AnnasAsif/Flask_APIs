from flask import Flask, render_template, request, jsonify,send_file, Response
from PIL import Image
from pymongo import MongoClient
import cv2
import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]

@app.route("/")
def index():
    print("Connected to MongoDB!")
    return render_template("index.html")

@app.route("/input")
def input():
    return render_template("input.html")

@app.route("/image", methods=["POST"])
def receive_image():
    try:
        image_data = request.files["image"]
        print(image_data)
        if not image_data:
            return jsonify({"error":"No image data"}), 400
        image_data.save("image.png")
        
        val=request.form["option"]
        if val == '1':
            img = Image.open("image.png")
            new_size = (400, 300)
            image = img.resize(new_size)
            image.save("newimage.png")
        elif val == '2':
            img = Image.open("image.png")
            image = img.rotate(45)
            image.save("newimage.png")
        elif val == '3':
            img = Image.open("image.png")
            width, height = img.size
            image = img.crop((width/4,height/4,(width/4)*3,(height/4)*3))
            image.save("newimage.png")
        elif val == '4':
            img = Image.open("image.png")
            image = img.transpose(Image.FLIP_TOP_BOTTOM)
            image.save("newimage.png")
        elif val == '5':
            img1 = cv2.imread("image.png", 0)
            _, thresholded = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)
            cv2.imwrite('newimage.png', thresholded)
        elif val == '6':
            img1 = cv2.imread("image.png")
            blurred = cv2.GaussianBlur(img1, (5, 5), 0)
            cv2.imwrite('newimage.png', blurred)
        elif val == '7':
            img1 = cv2.imread("image.png")
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(img1, -1, kernel)
            cv2.imwrite('newimage.png', sharpened)
        elif val == '8':
            img1 = cv2.imread("image.png")
            edges = cv2.Canny(img1, 100, 200)
            cv2.imwrite('newimage.png', edges)
        elif val == '9':
            img1 = cv2.imread("image.png")
            alpha = 1.5 # Simple contrast control
            beta = 0    # Simple brightness control
            enhanced = cv2.convertScaleAbs(img1, alpha=alpha, beta=beta)
            cv2.imwrite('newimage.png', enhanced)            
        elif val == '10':
            img1 = cv2.imread("image.png")
            Z = img1.reshape((-1,3))
            Z = np.float32(Z)

            # Define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 2
            ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

            # Now convert back into uint8, and make original image
            center = np.uint8(center)
            res = center[label.flatten()]
            segmented = res.reshape((img1.shape))
            cv2.imwrite('newimage.png', segmented)            
        elif val == '11':
            img1 = cv2.imread("image.png")
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('newimage.png', gray)            
        elif val == '12':
            img1 = cv2.imread("image.png")
            hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            cv2.imwrite('newimage.png', hsv)
            
        
        return send_file("newimage.png", mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/imageupload", methods=["POST"])
def upload_image():
    image = request.files["image"]
    print(type(request.form['option']))
    image.save("image.jpg")
    with open("image.jpg", "rb") as f:
        return Response(f.read(), content_type="image/jpeg")

# Background removal API
@app.route("/removebg", methods=["POST"])
def remove_bg():
    image = request.files["image"]
    image_name = image.filename
    save_directory = 'images/'

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    image.save(os.path.join(save_directory, image_name))

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("remove_bg")

    """ Loading model: DeepLabV3+ """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("model.h5")

    # model.summary()

    """ Load the dataset """
    data_x = glob("images/*")
    path = 'images/'+ image_name

    # for path in tqdm(data_x, total=len(data_x)):
    #      """ Extracting name """
    name = path.split("/")[-1].split(".")[0]

    """ Read the image """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
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

    cv2.imwrite(f"remove_bg/mask.png", photo_mask*255)
    # cv2.imwrite(f"remove_bg/{name}.png", background_mask*255)

    # cv2.imwrite(f"remove_bg/{name}.png", image * photo_mask)
    # cv2.imwrite(f"remove_bg/{name}.png", image * background_mask)

    masked_photo = image * photo_mask
    background_mask = np.concatenate([background_mask, background_mask, background_mask], axis=-1)
    background_mask = background_mask * [0, 0, 0]
    final_photo = masked_photo + background_mask
    temp = 'remove_bg/'+ name + '.jpg'
    print(temp)
    cv2.imwrite(temp, final_photo)

    # Replace 'static' with the path to your 'static' folder containing the images.
    # image_path = 'remove_bg/' + name + '.jpg'  # Replace with the name of your image file.
    
    processed_image_filename = name+ ".jpg"  # Replace with the actual processed filename.
    processed_image_path = os.path.join("remove_bg", processed_image_filename)

    # Replace backslashes with forward slashes in the path for the URL.
    processed_image_url = processed_image_path.replace("\\", "/")
    processed_image_url = f"http://172.16.0.94:9000/{processed_image_url}"

    # Return the URL or path of the processed image as a JSON response.
    response = {"processed_image_url": processed_image_url}
    return jsonify(response)
    # with open('remove_bg/'+ name + '.jpg', "rb") as f:
    #     return Response(f.read(), content_type="image/jpeg")


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
    app.run(host='172.16.0.94', port=9000, debug=True)