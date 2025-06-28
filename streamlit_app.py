import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# Load the model
@st.cache_resource
def load_cnn_model():
    model = load_model("Cnn_model.h5")
    return model

model = load_cnn_model()

# Set class names if you know them
class_names = ['normal', 'cataract', 'glaucoma', 'diabetic_retinopathy']  # Update based on your model

st.title("🧠 Image Classification with CNN")
st.write("Upload an image and the model will predict the class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    img_size = (256, 256)  # Update to your model's expected input size
    img_resized = img.resize(img_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize if required by your model

    # Make prediction
    prediction = model.predict(img_array)
    
    # Handle both binary and multi-class
    if prediction.shape[1] == 1:
        pred_class = int(prediction[0][0] > 0.5)
        confidence = prediction[0][0] if pred_class else 1 - prediction[0][0]
    else:
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)

    st.write(f"### 🏷️ Predicted Class: {class_names[pred_class]}")
    st.write(f"Confidence: {confidence:.2f}")
