# app.py
# Fake Currency Detection using GAN/CNN Model
# Streamlit Ready Full Code

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image
import cv2
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Fake Currency Detection",
    page_icon="💵",
    layout="wide"
)

# ---------------------------------------------------
# STYLE
# ---------------------------------------------------
st.markdown("""
<style>
.title{
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:#0b5394;
}
.sub{
    text-align:center;
    color:gray;
}
.result{
    font-size:28px;
    font-weight:bold;
}
.footer{
    text-align:center;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.markdown('<p class="title">💵 Fake Currency Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Upload Currency Note Image to Detect Genuine or Fake</p>', unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open("model_vgg.json", "r") as file:
            model_json = file.read()

        model = model_from_json(model_json)
        model.load_weights("model_vgg.weights.h5")

        return model

    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

model = load_model()

# ---------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------
classes = {
    0: "Fake Currency",
    1: "Real Currency"
}

# ---------------------------------------------------
# IMAGE PREPROCESS
# ---------------------------------------------------
def preprocess(img):

    img = img.resize((224,224))
    img = np.array(img)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------
def predict(img):

    processed = preprocess(img)

    pred = model.predict(processed)

    class_index = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    label = classes[class_index]

    return label, confidence, pred[0]

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("📌 Instructions")
st.sidebar.write("1. Upload currency image")
st.sidebar.write("2. Click Detect")
st.sidebar.write("3. View result")

st.sidebar.markdown("---")
st.sidebar.write("Supported Format:")
st.sidebar.write("JPG, JPEG, PNG")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Currency Note Image",
    type=["jpg","jpeg","png"]
)

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Currency Note", use_container_width=True)

    with col2:

        if st.button("🔍 Detect Currency"):

            if model is None:
                st.error("Model not loaded")

            else:
                with st.spinner("Analyzing Currency Note..."):

                    label, confidence, probs = predict(image)

                if label == "Real Currency":
                    st.success(f"✅ Prediction: {label}")
                else:
                    st.error(f"❌ Prediction: {label}")

                st.info(f"Confidence: {confidence:.2f}%")

                st.subheader("Prediction Score")

                chart_data = {
                    "Fake": float(probs[0]),
                    "Real": float(probs[1])
                }

                st.bar_chart(chart_data)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown('<p class="footer">Developed using Streamlit + Deep Learning</p>', unsafe_allow_html=True)
