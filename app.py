import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import json

# Page config
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    .real-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .fake-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .confidence-text {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('deepfake_detector.h5', compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_config():
    try:
        with open('model_config.json', 'r') as f:
            return json.load(f)
    except:
        return {'img_size': 224}

def preprocess_image(image, img_size):
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    img_array = cv2.resize(img_array, (img_size, img_size))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_deepfake(model, image, img_size):
    try:
        processed = preprocess_image(image, img_size)
        prediction = model.predict(processed, verbose=0)[0][0]
        return float(prediction)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    st.markdown('<h1 class="main-title">🔍 Deepfake Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Deepfake Detection System</p>', unsafe_allow_html=True)
    
    model = load_model()
    config = load_config()
    
    if model is None:
        st.error("⚠️ Model not loaded. Check if deepfake_detector.h5 is uploaded.")
        st.stop()
    
    st.success("✅ AI Model loaded!")
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("Upload a face image to detect if it's real or AI-generated.")
        st.divider()
        if 'test_accuracy' in config:
            st.metric("Model Accuracy", f"{config['test_accuracy']*100:.1f}%")
        st.write(f"**Model:** {config.get('model_type', 'MobileNetV2')}")
        st.divider()
        st.warning("⚠️ For screening purposes only")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 ANALYZE", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    prediction = predict_deepfake(model, image, config['img_size'])
                    if prediction is not None:
                        st.session_state['prediction'] = prediction
                        st.session_state['analyzed'] = True
                        st.rerun()
    
    with col2:
        st.subheader("📊 Results")
        
        if st.session_state.get('analyzed', False):
            prediction = st.session_state['prediction']
            is_fake = prediction > 0.5
            confidence = prediction if is_fake else (1 - prediction)
            
            result_class = "fake-box" if is_fake else "real-box"
            result_icon = "🚨" if is_fake else "✅"
            result_label = "FAKE" if is_fake else "REAL"
            result_desc = "AI-Generated" if is_fake else "Authentic"
            
            st.markdown(f"""
                <div class="result-box {result_class}">
                    <h2>{result_icon} {result_label}</h2>
                    <div class="confidence-text">{confidence*100:.1f}%</div>
                    <p style="font-size: 1.3rem;">{result_desc}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("### Breakdown")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Real", f"{(1-prediction)*100:.1f}%")
            with col_b:
                st.metric("Fake", f"{prediction*100:.1f}%")
            
            st.progress(float(confidence))
            
            if confidence >= 0.8:
                st.success("🟢 High Confidence")
            elif confidence >= 0.6:
                st.warning("🟡 Moderate Confidence")
            else:
                st.error("🔴 Low Confidence")
            
            if st.button("🔄 Analyze Another", use_container_width=True):
                st.session_state['analyzed'] = False
                st.rerun()
        else:
            st.info("Upload an image and click Analyze")
    
    st.divider()
    st.markdown("<div style='text-align: center; color: #666;'>Built with Streamlit & TensorFlow | Trained on Kaggle</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
