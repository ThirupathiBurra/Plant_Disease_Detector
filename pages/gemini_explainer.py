# ======================================
# 📌 Imports
# ======================================
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# ======================================
# 📌 Load Gemini API Key
# ======================================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ Google Gemini API key not found. Please set it in the .env file as GOOGLE_API_KEY.")
    st.stop()

genai.configure(api_key=api_key)

# ======================================
# 📌 Streamlit Setup
# ======================================
st.set_page_config(page_title="🌱 Disease Explanation", layout="centered")
st.title("🧠 Gemini Explanation for Plant Disease")

# ======================================
# 📌 Get predicted label
# ======================================
predicted_label = st.session_state.get('predicted_label', None)

if predicted_label:
    st.markdown(f"🔍 Disease detected: **{predicted_label}**")

    with st.spinner("Generating explanation using Gemini..."):
        prompt = f"""
        Give detailed information about the plant disease called {predicted_label}. 
        Include affected crops, causes, symptoms, prevention methods, and cure.
        Keep the explanation beginner-friendly and concise.
        """
        try:
            # ✅ Correct model name based on Gemini API
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content(prompt)

            # ✅ Safely extract text from Gemini response
            if hasattr(response, 'text'):
                st.markdown(response.text)
            else:
                st.error("⚠️ Gemini response is empty. Try again later.")

        except Exception as e:
            st.error(f"❌ Gemini Error: {e}")
else:
    st.warning("⚠️ No prediction found. Please go back and predict first.")
