from flask import Flask, render_template, request, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


app = Flask(__name__)

# Specify the directory where the model is saved
model_dir = './model'
model_path = os.path.join(model_dir, 'fashion_mnist_model.h5')

# Load the pre-trained model
model = load_model(model_path)

# Directory to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to process uploaded image and make a prediction
def predict_category(img_path):
    img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
        
        # Check if the file has a valid filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        # ... (rest of your code for processing the uploaded image and making predictions)
        
        # Get the filename of the uploaded image
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Make a prediction
        predicted_class = predict_category(filename)
        
        # Get the class label from Fashion MNIST dataset
        class_labels = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        prediction_label = class_labels[predicted_class]
        
        # Display the prediction result and uploaded image
        return render_template('index.html', message='Prediction: {}'.format(prediction_label), image_path=filename, os=os)
    
    return render_template('index.html', message='Upload an image', os=os)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)