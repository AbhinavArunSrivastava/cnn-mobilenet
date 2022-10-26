from flask import Flask, request, jsonify, url_for, render_template
import uuid #unique id so that it can be used as a identifier for predictions
import os


Allowed_extension = set(["png", "jpg", "jpeg", "gif" ])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".")[1] in Allowed_extension

app = Flask(__name__) #object : resources: html, css, js and other files

@app.route('/')
def index():
    return render_template('ImageML.html')


import numpy as np
from werkzeug.utils import secure_filename

#CNN mobileNet model
from PIL import Image, ImageFile

from io import BytesIO

#model 
from tensorflow.keras.applications import MobileNet  #cnn model

#preprocessing libraries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions



model = MobileNet(weights = "imagenet", include_top = True)


model.summary()

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNEL = 3


@app.route('/api/image', methods = ["POST"])
def upload_image():
    if 'image' not in request.files:
        return render_template("ImageML.html", prediction = "No Posted Image")
    file = request.files["image"] #else part 
    
    if file.filename == '':
        return render_template("ImageML.html", prediction = "You did not select an image")
    
    #correcly solved then we say that file is sucessful to be uploaded
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        x = [] # numpy array containing pixels of image
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        
        #load image here
        img = Image.open(BytesIO(file.read()))
        img.load()
        
        #This is all to be done for model
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        
        #numpy array
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        
        x = preprocess_input(x) #input is ready
        
        pred = model.predict(x)
        lst = decode_predictions(pred, top = 3 )
        
        items = []
        for item in lst[0]:
            items.append({"name": item[1], 'prob': float(item[2])})
            
        response = {'pred': items}
        return render_template('ImageML.html', prediction = f'My Prediction: {response["pred"]}')
    
    else:
        return render_template("ImageML.html", prediction = "Invalid file extension")
    
    
    if __name__ == '__main__':
    app.run(debug=True, use_reloader = False)