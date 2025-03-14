import streamlit as st
import numpy as np
import cv2
import joblib
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load models
cnn_model = load_model("cnn_malaria_model.h5")
rf_model = joblib.load("random_forest_malaria.pkl")

# Load SHAP Explainers
rf_explainer = shap.Explainer(rf_model)
cnn_explainer = None  # CNN SHAP needs DeepExplainer

# Image size for CNN model
IMG_SIZE = 64

# Streamlit UI
st.title("🦠 Malaria Detection App")
st.write("Upload a **cell image** to predict Malaria")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize

    # Display image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # CNN Prediction
    img_array = np.expand_dims(img, axis=0)
    pred_cnn = cnn_model.predict(img_array)[0][0]

    st.subheader("🧪 CNN Malaria Probability")
    st.write(f"**{round(float(pred_cnn), 2)}**")

    # SHAP for CNN
    global cnn_explainer
    if cnn_explainer is None:
        cnn_explainer = shap.GradientExplainer(cnn_model, img_array)

    shap_values = cnn_explainer.shap_values(img_array)

    # Plot SHAP
    st.subheader("📊 SHAP Explanation (CNN)")
    plt.figure()
    shap.image_plot(shap_values, img_array, show=False)
    st.pyplot(plt)
