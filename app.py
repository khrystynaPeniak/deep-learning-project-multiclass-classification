import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input


MODEL_PATH = "/content/drive/MyDrive/mashrooms-vision/model-b3-100.keras"
DATASET_PATH = "/content/drive/MyDrive/mashrooms-vision/dataset/dataset/"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = [str(i) for i in range(10)]

st.title("🍄 Mushroom Classifier")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        x = img.resize((224,224))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)[0]
        idx = int(np.argmax(preds))
        st.subheader(f"Prediction: class **{class_names[idx]}**")
        st.write("Confidence:")
        for name, prob in zip(class_names, preds):
            st.write(f"{name}: {prob:.4f}")
