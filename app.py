import streamlit as st
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

# Load model with better error handling
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        # Suppress TF warnings
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        model = keras.models.load_model('deepfake_detector.h5', compile=False)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_config():
    try:
        with open('model_config.json', 'r') as f:
            return json.load(f)
    except:
        return {'img_size': 224}

def preprocess_image(image, img_size):
    """Preprocess image for model"""
    img_array = np.array(image)
    
    # Convert to RGB
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize
    img_array = cv2.resize(img_array, (img_size, img_size))
    
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_deepfake(model, image, img_size):
    """Make prediction"""
    try:
        processed = preprocess_image(image, img_size)
        prediction = model.predict(processed, verbose=0)[0][0]
        return float(prediction), None
    except Exception as e:
        return None, str(e)

def main():
    # Header
    st.markdown('<h1 class="main-title">🔍 Deepfake Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Deepfake Detection System</p>', unsafe_allow_html=True)
    
    # Load config
    config = load_config()
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model, error = load_model()
    
    if model is None:
        st.error("⚠️ **Model Loading Failed**")
        st.error(f"Error: {error}")
        st.info("""
        **Troubleshooting:**
        - Ensure `deepfake_detector.h5` is uploaded to GitHub
        - Check if file size is under 100MB
        - Try using Git LFS for large files
        """)
        st.stop()
    
    st.success("✅ AI Model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        Upload a face image to detect if it's real or AI-generated.
        
        **How to use:**
        1. Upload image
        2. Click Analyze
        3. View results
        """)
        
        st.divider()
        
        st.write("**Model Info:**")
        if 'test_accuracy' in config:
            st.metric("Accuracy", f"{config['test_accuracy']*100:.1f}%")
        
        st.write(f"**Type:** {config.get('model_type', 'MobileNetV2')}")
        st.write(f"**Size:** {config.get('img_size', 224)}×{config.get('img_size', 224)}")
        
        st.divider()
        
        st.warning("⚠️ For screening purposes only. Not definitive proof.")
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a face photo"
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)
                
                with st.expander("Image Details"):
                    st.write(f"**Size:** {image.size[0]} × {image.size[1]} px")
                    st.write(f"**Format:** {image.format}")
                    st.write(f"**Mode:** {image.mode}")
                
                # Analyze button
                col_a, col_b, col_c = st.columns([1, 2, 1])
                with col_b:
                    analyze = st.button("🔍 ANALYZE IMAGE", type="primary", use_container_width=True)
                
                if analyze:
                    with st.spinner("🧠 AI is analyzing..."):
                        prediction, pred_error = predict_deepfake(model, image, config['img_size'])
                        
                        if prediction is not None:
                            st.session_state['prediction'] = prediction
                            st.session_state['analyzed'] = True
                            st.rerun()
                        else:
                            st.error(f"Prediction failed: {pred_error}")
            
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    with col2:
        st.subheader("📊 Results")
        
        if st.session_state.get('analyzed', False):
            prediction = st.session_state['prediction']
            
            # Calculate results
            is_fake = prediction > 0.5
            confidence = prediction if is_fake else (1 - prediction)
            
            # Display result
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
            
            # Detailed breakdown
            st.write("### 📈 Probability Breakdown")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Real Probability", f"{(1-prediction)*100:.1f}%")
            with col_b:
                st.metric("Fake Probability", f"{prediction*100:.1f}%")
            
            st.write("**Confidence Level:**")
            st.progress(float(confidence))
            
            # Confidence interpretation
            if confidence >= 0.85:
                st.success("🟢 **Very High Confidence** - Strong detection")
            elif confidence >= 0.70:
                st.info("🟡 **High Confidence** - Reliable result")
            elif confidence >= 0.55:
                st.warning("🟠 **Moderate Confidence** - Consider with caution")
            else:
                st.error("🔴 **Low Confidence** - Uncertain result")
            
            if confidence < 0.7:
                st.warning("""
                ⚠️ **Low Confidence Alert**
                
                Possible reasons:
                - Poor image quality
                - Unusual lighting
                - Partial face visibility
                - Non-standard angle
                """)
            
            st.info("💡 **Remember:** This is an AI screening tool. Always verify important findings through additional means.")
            
            # Reset button
            if st.button("🔄 Analyze Another Image", use_container_width=True):
                st.session_state['analyzed'] = False
                del st.session_state['prediction']
                st.rerun()
        
        else:
            # Empty state
            st.info("📥 **Ready to analyze**\n\nUpload an image on the left and click 'Analyze'")
            
            st.write("### ✨ Features")
            st.write("⚡ **Instant** - Results in seconds")
            st.write("🎯 **Accurate** - Trained on 1000s of images")
            st.write("🔒 **Private** - All processing done securely")
            st.write("🆓 **Free** - No usage limits")
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; padding: 1rem; color: #666;'>
            <p style='margin: 0; font-size: 1rem;'><strong>Deepfake Detector</strong></p>
            <p style='margin: 0; font-size: 0.9rem;'>Built with Streamlit & TensorFlow | Trained on Kaggle</p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #999;'>Educational & Research Use Only</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
