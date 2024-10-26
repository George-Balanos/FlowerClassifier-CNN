from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('final_model.h5')

class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

def preprocess_image(image_bytes):
    # Open the image and resize to the model’s input size
    image = Image.open(io.BytesIO(image_bytes)).resize((128, 128))
    image = np.array(image) / 255.0  # Normalize pixel values if needed
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def home():
    return render_template('index.html')  # Renders index.html when the home route is accessed


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Decode the base64 image
    image_data = base64.b64decode(data['image'])
    image = preprocess_image(image_data)

    # Make prediction
    prediction = model.predict(image)

    # Get the index of the highest probability
    predicted_index = np.argmax(prediction[0])

    # Get the corresponding class label
    predicted_class = class_names[predicted_index]

    return jsonify({'prediction': predicted_class})
if __name__ == '__main__':
    app.run(debug=True)
