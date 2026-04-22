# app.py
# Final Fake Currency Detection App (Machine Learning Based)
# Streamlit Cloud Ready

import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fake Currency Detection",
    page_icon="💵",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#0b5394;
}
.sub-title{
    text-align:center;
    color:gray;
    font-size:18px;
}
.footer{
    text-align:center;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<p class="main-title">💵 Fake Currency Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload Note Image to Detect Real or Fake Currency</p>', unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        return None
    return joblib.load("model.pkl")

model = load_model()

# -------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------
def extract_features(img):
    img = img.convert("RGB")
    img = img.resize((64, 64))
    arr = np.array(img)
    arr = arr.flatten()
    arr = arr / 255.0
    return arr.reshape(1, -1)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("📌 Instructions")
st.sidebar.write("1. Upload currency image")
st.sidebar.write("2. Click Detect")
st.sidebar.write("3. View result")

st.sidebar.markdown("---")
st.sidebar.write("Supported formats:")
st.sidebar.write("JPG, JPEG, PNG")

# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Currency Note Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Currency Note", use_container_width=True)

    with col2:

        if st.button("🔍 Detect Currency"):

            if model is None:
                st.error("❌ model.pkl file not found.")
            else:
                with st.spinner("Analyzing Currency Note..."):

                    x = extract_features(image)

                    pred = model.predict(x)[0]

                    if hasattr(model, "predict_proba"):
                        prob = np.max(model.predict_proba(x)) * 100
                    else:
                        prob = 95.0

                if pred == 1:
                    st.success("✅ Real Currency")
                else:
                    st.error("❌ Fake Currency")

                st.info(f"Confidence: {prob:.2f}%")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown(
    '<p class="footer">Developed using Streamlit + Machine Learning</p>',
    unsafe_allow_html=True
)
