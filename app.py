# app.py
import os
import streamlit as st
import gdown
from PIL import Image
import torch
from model import load_model, predict_image, DEVICE

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI vs Real Image Detector", layout="centered")
st.title("🤖 AI vs Real Image Detector")
st.write("Upload an image and the model will predict whether it is AI-generated or real.")

# =========================
# MODEL DOWNLOAD
# =========================
MODEL_PATH = "best_model.pth"
FILE_ID = "1xiaaY4kTfztkKd0I04DmFKlASikRLhN5"  # Replace with your Google Drive File ID
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("⬇️ Downloading model from Google Drive..."):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def get_model():
    model = load_model(MODEL_PATH)
    return model

model = get_model()

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    if st.button("Predict"):
        with st.spinner("🔍 Predicting..."):
            pred, conf, probs = predict_image(model, image)

            ai_prob = probs[0]
            real_prob = probs[1]

            if pred == 0:
                label = "🤖 AI Generated"
            else:
                label = "📷 Real Image"

            st.success(f"Prediction: {label}")
            st.write(f"Confidence: {conf*100:.2f}%")
            st.write("### Probabilities")
            st.write(f"AI: {ai_prob*100:.2f}%")
            st.write(f"Real: {real_prob*100:.2f}%")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & PyTorch")