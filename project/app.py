import os
import io
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request, render_template, redirect
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load multiple saved models including custom_cancer
def load_saved_models(model_names, model_save_path='new_saved_models/'):
    models = {}
    for model_name in model_names:
        model_path = os.path.join(model_save_path, f'{model_name}.keras')
        model = tf.keras.models.load_model(model_path)
        print(f"{model_name} model loaded successfully from {model_path}")
        models[model_name] = model
    return models

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the input image
def preprocess_image(image):
    img = image.resize((150, 150))  # Resize the image to the model's input size
    img_array = img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make predictions
def predict_image(model, image, class_names):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100
    
    if confidence < 90:
        predicted_class = "No disease found"
    else:
        predicted_class = class_names[predicted_class_index]
        predicted_class = predicted_class.replace(' Augmented', '')  # Remove augmented if present
    
    return predicted_class, confidence

# Encode image to Base64
def encode_image_to_base64(image):
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')
    return img_base64

# Define class names (replace with your actual class names)
class_names = [
    'Acral Lentiginous Melanoma Augmented', 'Alopecia Areata Augmented', 'Alopecia Totalis Augmented',
    'Androgenetic Alopecia Augmented', 'Arsenicosis Augmented', 'Basal Cell Carcinoma Augmented', 
    'Bowens Disease Augmented', 'Chromoblastomycosis Augmented', 'Dariers Disease Augmented', 
    'Discoid Lupus Erythematosus Augmented', 'Drug Eruptions Augmented', 'Drug Reactions Augmented',
    'Ecthyma Augmented', 'Epidermolytic Hyperkeratosis Augmented', 'Fordyce Spots Augmented', 
    'Granuloma Annulare Augmented', 'Granulomatous Diseases Augmented', 'Hemangioma Augmented', 
    'Herpes Zoster Augmented', 'Hypertrophic Lichen Planus Augmented', 'Ichthyosis Augmented',
    'Impetigo Contagiosa Augmented', 'Keratoderma Augmented', 'Lichen Planus Augmented', 
    'Linear Scleroderma Augmented', 'Livedo Reticularis Augmented', 'Lupus Vulgaris Augmented', 
    'Malignant Acanthosis Nigricans Augmented', 'Malignant Melanoma Augmented', 'Melanoacanthoma Augmented',
    'Mole Augmented', 'Molluscum Contagiosum Augmented', 'Nevus Augmented', 'Nevus of Ota Augmented', 
    'Nevus Sebaceus Augmented', 'Nevus Spilus Augmented', 'Oral Lichen Planus Augmented', 
    'Paget_s Disease Augmented', 'Pemphigus Vulgaris Augmented', 'Pityriasis Lichenoides Chronica Augmented',
    'Pityriasis Rosea Augmented', 'Pityriasis Versicolor Augmented', 'Psoriasis Augmented', 
    'Pyogenic Granuloma Augmented', 'Seborrheic Keratosis Augmented', 'Solitary Mastocytosis Augmented',
    'Squamous cell carcinoma Augmented', 'Striae Distensae Augmented', 'Systemic Lupus Erythematosus Augmented', 
    'Tinea Barbae Augmented', 'Tinea Corporis Augmented', 'Tinea Faciei Augmented', 'Tinea Pedis Augmented', 
    'Trichoepithelioma Augmented', 'Tuberculosis Verrucosa Cutis Augmented', 'Verruca Augmented', 
    'Vitiligo Augmented'
]

# Load the models including custom_cancer
model_names = ['InceptionV3', 'custom_model']
models = load_saved_models(model_names)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Open the image using PIL
        image = Image.open(file)

        # Predict the class of the image using all models including custom_cancer
        results = {}
        for model_name, model in models.items():
            predicted_class, confidence = predict_image(model, image, class_names)
            results[model_name] = {
                'predicted_class': predicted_class,
                'confidence': confidence
            }

        # Encode the image to Base64
        img_base64 = encode_image_to_base64(image)

        # Pass the Base64 image and prediction result for all models to the template
        return render_template('index.html', img_data=img_base64, results=results)

    return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)
