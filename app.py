# ======================================
# 📌 Imports
# ======================================
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from utils.image_utils import predict

# ======================================
# 📌 Load environment variables
# ======================================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ Google Gemini API key not found. Please set it in the .env file.")
    st.stop()

genai.configure(api_key=api_key)

# ======================================
# 📌 Load class indices
# ======================================
with open('model/class_indices.json') as f:
    class_indices = json.load(f)
idx2label = {v: k for k, v in class_indices.items()}

# ======================================
# 📌 Load TFLite model path
# ======================================
model_path = os.path.join("model", "plant_disease_model.tflite")

# ======================================
# 📌 Preprocess image (PIL version)
# ======================================
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((224, 224))  # Model input size
    image = np.array(image).astype(np.float32) / 255.0
    return image

# ======================================
# 📌 Streamlit App UI
# ======================================
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 Plant Leaf Disease Detector")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = preprocess_image(uploaded_file)

    # Get prediction (returns label, confidence, about, treatment)
    label, confidence, description, treatment = predict(
        image,
        model_path=model_path,
        idx2label=idx2label,
        disease_info_path='disease_info.json'
    )
    confidence_percent = float(confidence) * 100

    # Display results
    st.success(f"✅ Prediction: {label} ({confidence_percent:.2f}% confidence)")
    st.markdown(f"**About:** {description}")
    st.markdown(f"**Treatment:** {treatment}")

    # 🔍 GPT Button
    if st.button("🔍 Learn more about this disease"):
        st.session_state['predicted_label'] = label
        st.switch_page("pages/gemini_explainer.py")

    # 📄 Report download
    report = f"""
📋 **Leaf Disease Report**

**Prediction:** {label}
**Confidence:** {confidence_percent:.2f}%
**About:** {description}
**Treatment:** {treatment}
"""
    st.download_button(
        label="📄 Download Report",
        data=report.encode("utf-8"),
        file_name=f"{label}_report.txt",
        mime="text/plain"
    )