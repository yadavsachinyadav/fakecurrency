# app.py
# Streamlit Cloud Optimized Fake Currency Detection

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Fake Currency Detection",
    page_icon="💵",
    layout="wide"
)

# -------------------------------
# SAFE TENSORFLOW IMPORT
# -------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import model_from_json
except Exception as e:
    st.error("TensorFlow not installed correctly.")
    st.stop()

# -------------------------------
# CUSTOM CSS
# -------------------------------
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
font-size:18px;
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

# -------------------------------
# TITLE
# -------------------------------
st.markdown('<p class="title">💵 Fake Currency Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Upload Note Image to Check Real or Fake</p>', unsafe_allow_html=True)

# -------------------------------
# MODEL LOAD
# -------------------------------
@st.cache_resource
def load_model():
    try:
        if not os.path.exists("model_vgg.json"):
            st.error("model_vgg.json file missing")
            return None

        if not os.path.exists("model_vgg.weights.h5"):
            st.error("model_vgg.weights.h5 file missing")
            return None

        with open("model_vgg.json", "r") as f:
            model_json = f.read()

        model = model_from_json(model_json)
        model.load_weights("model_vgg.weights.h5")

        return model

    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None

model = load_model()

# -------------------------------
# LABELS
# -------------------------------
classes = {
    0: "Fake Currency",
    1: "Real Currency"
}

# -------------------------------
# IMAGE PROCESSING
# -------------------------------
def preprocess(img):

    img = img.resize((224, 224))
    img = np.array(img)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# -------------------------------
# PREDICTION
# -------------------------------
def predict(img):

    processed = preprocess(img)

    pred = model.predict(processed, verbose=0)

    class_index = np.argmax(pred)
    confidence = np.max(pred) * 100

    return classes[class_index], confidence, pred[0]

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload note image")
st.sidebar.write("2. Click Detect")
st.sidebar.write("3. See Result")

# -------------------------------
# FILE UPLOADER
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Currency Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# MAIN
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Note", use_container_width=True)

    with col2:

        if st.button("🔍 Detect Currency"):

            if model is None:
                st.error("Model not loaded.")
            else:
                with st.spinner("Checking Note..."):

                    label, confidence, probs = predict(image)

                if label == "Real Currency":
                    st.success(f"✅ {label}")
                else:
                    st.error(f"❌ {label}")

                st.info(f"Confidence: {confidence:.2f}%")

                st.subheader("Prediction Score")

                chart = {
                    "Fake": float(probs[0]),
                    "Real": float(probs[1])
                }

                st.bar_chart(chart)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    '<p class="footer">Developed with Streamlit + TensorFlow</p>',
    unsafe_allow_html=True
)
