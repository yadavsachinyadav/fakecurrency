import streamlit as st
import numpy as np
from PIL import Image
import os

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Fake Currency Detection",
    page_icon="💵",
    layout="wide"
)

# --------------------------------
# LOAD TENSORFLOW
# --------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import model_from_json
except:
    st.error("TensorFlow not installed.")
    st.stop()

# --------------------------------
# TITLE
# --------------------------------
st.title("💵 Fake Currency Detection")
st.write("Upload note image to check Real / Fake")

# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    try:
        with open("model_vgg.json", "r") as f:
            model_json = f.read()

        model = model_from_json(model_json)
        model.load_weights("model_vgg.weights.h5")

        return model
    except Exception as e:
        st.error(e)
        return None

model = load_model()

# --------------------------------
# CLASS LABELS
# --------------------------------
classes = {
    0: "Fake Currency",
    1: "Real Currency"
}

# --------------------------------
# IMAGE PREPROCESS
# --------------------------------
def preprocess(img):

    img = img.convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# --------------------------------
# PREDICT
# --------------------------------
def predict(img):

    processed = preprocess(img)

    pred = model.predict(processed, verbose=0)

    index = np.argmax(pred)
    confidence = np.max(pred) * 100

    return classes[index], confidence

# --------------------------------
# FILE UPLOAD
# --------------------------------
file = st.file_uploader("Upload Currency Image", type=["jpg", "jpeg", "png"])

if file:

    image = Image.open(file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Currency"):

        label, conf = predict(image)

        if label == "Real Currency":
            st.success(f"✅ {label}")
        else:
            st.error(f"❌ {label}")

        st.info(f"Confidence: {conf:.2f}%")
