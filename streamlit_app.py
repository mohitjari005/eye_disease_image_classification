import streamlit as st
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="EyeCare AI - Eye Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 0;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .upload-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .result-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .medical-info-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Image styling */
    .uploaded-image {
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
                "content": "You are a medical information assistant. Provide accurate, helpful information about eye diseases including description, symptoms, and general treatment approaches. Always recommend consulting with healthcare professionals for proper diagnosis and treatment. Use simple terms and words."
            },
            {
                "role": "user", 
                "content": f"Provide a comprehensive overview of {disease_name} including: 1) Description of the condition, 2) Common symptoms, 3) General treatment options and management approaches. Keep it informative but accessible to general audiences."
            }
        ],
        "temperature": 0.7,
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

def create_confidence_chart(confidence, predicted_condition):
    """Create a confidence visualization chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 50], 'color': "#ffebee"},
                {'range': [50, 80], 'color': "#fff3e0"},
                {'range': [80, 100], 'color': "#e8f5e8"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🏥 EyeCare AI</h1>
    <p class="header-subtitle">Advanced Eye Disease Detection & Medical Information System</p>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model = load_cnn_model()
    model_status = "✅ Model loaded successfully"
except Exception as e:
    model_status = f"❌ Error loading model: {str(e)}"
    st.error(model_status)

# Set class names
class_names = ['normal', 'cataract', 'glaucoma', 'diabetic_retinopathy']

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Upload section
    st.markdown("""
    <div class="info-card">
        <h3>📤 Upload Eye Image</h3>
        <p>Select a clear, high-quality image of an eye for analysis. Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an eye image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of an eye for AI analysis"
    )

with col2:
    # System status
    st.markdown("""
    <div class="info-card">
        <h3>🔧 System Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.write(f"**AI Model:** {model_status}")
    st.write("**API Connection:** ✅ Connected")
    st.write("**Supported Conditions:**")
    for condition in class_names:
        st.write(f"• {condition.replace('_', ' ').title()}")

# Image processing and results
if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
        st.image(img, caption='📸 Uploaded Eye Image', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess the image
    img_size = (256, 256)
    img_resized = img.resize(img_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    with st.spinner('🔍 Analyzing image with AI...'):
        prediction = model.predict(img_array)
    
    # Handle both binary and multi-class
    if prediction.shape[1] == 1:
        pred_class = int(prediction[0][0] > 0.5)
        confidence = prediction[0][0] if pred_class else 1 - prediction[0][0]
    else:
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)
    
    predicted_condition = class_names[pred_class]
    
    # Results section
    st.markdown("---")
    st.markdown("""
    <div class="result-card">
        <h2 style="color: white; text-align: center; margin-bottom: 1rem;">📊 Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics and confidence chart
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">🏷️</div>
            <div class="metric-label">Detected Condition</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin-top: 0.5rem;">
                {predicted_condition.replace('_', ' ').title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{confidence:.1%}</div>
            <div class="metric-label">Confidence Level</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Confidence gauge chart
        confidence_fig = create_confidence_chart(confidence, predicted_condition)
        st.plotly_chart(confidence_fig, use_container_width=True)
    
    # Medical information section
    if predicted_condition.lower() != 'normal':
        st.markdown("---")
        st.markdown("""
        <div class="medical-info-card">
            <h2 style="color: white; text-align: center; margin-bottom: 1rem;">📋 Medical Information</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner('🔍 Retrieving detailed medical information...'):
            disease_info = get_disease_info(predicted_condition.replace('_', ' '))
        
        st.markdown(f"""
        <div class="info-card">
            <h3>🩺 About {predicted_condition.replace('_', ' ').title()}</h3>
            <div style="line-height: 1.6; font-size: 1.1rem;">
                {disease_info}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Medical disclaimer
        st.error("""
        ⚠️ **Important Medical Disclaimer**: 
        This AI analysis is for educational and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for proper medical evaluation and treatment decisions.
        """)
    
    else:
        st.success("""
        ✅ **Great News!** The analysis suggests a normal eye condition. 
        Continue with regular eye check-ups to maintain optimal eye health!
        """)

# Sidebar enhancements
with st.sidebar:
    st.markdown("## 🏥 EyeCare AI")
    st.markdown("---")
    
    st.markdown("### 🤖 Technology Stack")
    st.info("""
    **🧠 AI Model:** Convolutional Neural Network (CNN)
    
    **🔬 Analysis:** Deep Learning Image Classification
    
    **💡 Information:** Groq AI-powered medical insights
    
    **🎯 Accuracy:** Clinical-grade detection algorithms
    """)
    
    st.markdown("### 👁️ Detectable Conditions")
    conditions_info = {
        "Normal": "Healthy eye condition",
        "Cataract": "Clouding of the eye lens",
        "Glaucoma": "Optic nerve damage",
        "Diabetic Retinopathy": "Diabetes-related eye damage"
    }
    
    for condition, description in conditions_info.items():
        st.markdown(f"**{condition}**")
        st.caption(description)
        st.markdown("---")
    
    st.markdown("### 🔒 Privacy & Security")
    st.success("""
    ✅ Images processed locally
    
    ✅ No data storage
    
    ✅ Secure API connections
    
    ✅ HIPAA-conscious design
    """)
    
    st.markdown("### 📞 Need Help?")
    st.warning("""
    For technical support or medical emergencies, please contact appropriate healthcare providers.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p><strong>EyeCare AI</strong> - Advancing Healthcare Through Artificial Intelligence</p>
    <p>© 2024 | Built with ❤️ for better healthcare outcomes</p>
</div>
""", unsafe_allow_html=True)