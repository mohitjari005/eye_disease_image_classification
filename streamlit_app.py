import streamlit as st
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# Groq API configuration
GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Load the model
@st.cache_resource
def load_cnn_model():
    model = load_model("Cnn_model.h5")
    return model

def get_disease_info(disease_name):
    """Get disease description and treatment from Groq API"""
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system", 
                "content": "You are a medical information assistant. Provide accurate, helpful information about eye diseases including description, symptoms, and general treatment approaches. Always recommend consulting with healthcare professionals for proper diagnosis and treatment."
            },
            {
                "role": "user", 
                "content": f"Provide a comprehensive overview of {disease_name} including: 1) Description of the condition, 2) Common symptoms, 3) General treatment options and management approaches. Keep it informative but accessible to general audiences."
            }
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"Error getting information: {response.status_code}"
    except Exception as e:
        return f"Error connecting to API: {str(e)}"

model = load_cnn_model()

# Set class names
class_names = ['normal', 'cataract', 'glaucoma', 'diabetic_retinopathy']

st.title("👁️ Eye Disease Classification & Information System")
st.write("Upload an eye image and get AI-powered classification with detailed medical information.")

# Upload image
uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    img_size = (256, 256)
    img_resized = img.resize(img_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    with st.spinner('Analyzing image...'):
        prediction = model.predict(img_array)
    
    # Handle both binary and multi-class
    if prediction.shape[1] == 1:
        pred_class = int(prediction[0][0] > 0.5)
        confidence = prediction[0][0] if pred_class else 1 - prediction[0][0]
    else:
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)
    
    predicted_condition = class_names[pred_class]
    
    # Display prediction results
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🏷️ Predicted Condition", predicted_condition.replace('_', ' ').title())
    with col2:
        st.metric("📊 Confidence", f"{confidence:.1%}")
    
    # Get detailed information if not normal
    if predicted_condition.lower() != 'normal':
        st.markdown("---")
        st.markdown("### 📋 Medical Information")
        
        with st.spinner('Getting detailed medical information...'):
            disease_info = get_disease_info(predicted_condition.replace('_', ' '))
        
        st.markdown(disease_info)
        
        # Add disclaimer
        st.warning("⚠️ **Medical Disclaimer**: This information is for educational purposes only and should not replace professional medical advice. Please consult with an eye care professional for proper diagnosis and treatment.")
    
    else:
        st.success("✅ The analysis suggests normal eye condition. Continue regular eye check-ups to maintain good eye health!")

# Add sidebar with information
st.sidebar.markdown("### ℹ️ About This App")
st.sidebar.write("""
This application uses:
- **CNN Model**: For eye disease classification
- **Groq AI**: For detailed medical information
- **Conditions Detected**: Normal, Cataract, Glaucoma, Diabetic Retinopathy
""")

st.sidebar.markdown("### 🔒 Privacy Note")
st.sidebar.write("Images are processed locally and not stored or transmitted except for AI analysis.")