import streamlit as st
import numpy as np
import pandas as pd
import shap
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt

# Load Models
cnn_model = load_model("models/malaria_cnn.h5")  # CNN Model
rf_model = joblib.load("models/random_forest.pkl")  # Random Forest
logistic_model = joblib.load("models/logistic_regression.pkl")  # Logistic Regression

# Load Scaler
scaler = joblib.load("models/scaler.pkl")  # StandardScaler for clinical data

# Initialize SHAP Explainer (Global Variable)
cnn_explainer = None

# Streamlit UI
st.title("Malaria Prediction App")
st.write("Upload clinical data and cell image to predict malaria.")

# Select Model
model_choice = st.selectbox(
    "Choose a Model:",
    ["Random Forest", "Logistic Regression", "CNN + LSTM"]
)

# Upload Clinical Data
clinical_file = st.file_uploader("Upload Clinical Data (CSV)", type=["csv"])

# Upload Image
image_file = st.file_uploader("Upload Cell Image (JPG/PNG)", type=["jpg", "png"])

# Process Clinical Data
def preprocess_clinical_data(data):
    data = data.fillna(data.median())  # Handle missing values
    categorical_cols = ["Gender", "Fever", "BodyPain"]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    data_scaled = scaler.transform(data)
    return data_scaled

# Process Image
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)  # Read image
    img = cv2.resize(img, (64, 64)) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

if st.button("Predict"):
    if not clinical_file or not image_file:
        st.error("Please upload both clinical data and an image.")
    else:
        # Read Clinical Data
        clinical_data = pd.read_csv(clinical_file)
        X_clinical = preprocess_clinical_data(clinical_data)

        # Read and Process Image
        X_image = preprocess_image(image_file)

        # Perform Prediction
        if model_choice == "Random Forest":
            prediction = rf_model.predict(X_clinical)[0]
            explanation = "Random Forest uses decision trees for classification."
        elif model_choice == "Logistic Regression":
            prediction = logistic_model.predict(X_clinical)[0]
            explanation = "Logistic Regression is a statistical model for binary classification."
        elif model_choice == "CNN + LSTM":
            cnn_pred = cnn_model.predict(X_image)[0][0]
            prediction = 1 if cnn_pred > 0.5 else 0  # Convert probability to label
            explanation = "CNN + LSTM predicts malaria based on image features."

            # SHAP Explanation (if CNN selected)
            global cnn_explainer
            if cnn_explainer is None:
                cnn_explainer = shap.Explainer(cnn_model, X_image)
            shap_values = cnn_explainer(X_image)

            # Show SHAP Explanation
            st.subheader("SHAP Explanation for CNN Model")
            fig, ax = plt.subplots()
            shap.image_plot(shap_values, X_image, show=False)
            st.pyplot(fig)

        # Show Prediction
        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {'Positive (Malaria)' if prediction == 1 else 'Negative (No Malaria)'}")
        st.write(f"**Model Explanation:** {explanation}")
