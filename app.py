import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import joblib

# Initialize global variables
cnn_explainer = None  
cnn_model = None  
rf_model = None  
log_reg_model = None  

# Set Streamlit page config
st.set_page_config(page_title="Malaria Detection", layout="wide")

# Title
st.title("🦠 Malaria Detection & Prediction System")

# Sidebar for file upload
st.sidebar.header("Upload Data")
clinical_file = st.sidebar.file_uploader("Upload Clinical Data (CSV)", type=["csv"])
image_dir = st.sidebar.text_input("Enter Path to Malaria Images")

# Load clinical data
if clinical_file:
    clinical_data = pd.read_csv(clinical_file)
    st.write("### Clinical Data Preview", clinical_data.head())

    # Preprocessing
    st.write("### Clinical Data Preprocessing")
    clinical_data = pd.get_dummies(clinical_data, drop_first=True)
    st.write(clinical_data.head())

# Load malaria images
IMG_SIZE = 64

def load_images(image_dir):
    images, labels = [], []
    if image_dir:
        for category in ["Parasitized", "Uninfected"]:
            path = f"{image_dir}/{category}"
            label = 1 if category == "Parasitized" else 0
            for img_name in os.listdir(path)[:100]:  # Limit to 100 for speed
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
    return np.array(images) / 255.0, np.array(labels)

if image_dir:
    X_images, y_images = load_images(image_dir)
    st.write(f"Loaded {len(X_images)} images.")

# Model Selection
st.sidebar.header("Select Model")
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "CNN + LSTM"])

# Train/Test Split
if clinical_file:
    X_clinical = clinical_data.drop("malaria_test_result", axis=1)
    y_clinical = clinical_data["malaria_test_result"]
    
    X_train, X_test, y_train, y_test = train_test_split(X_clinical, y_clinical, test_size=0.2, random_state=42)

    if model_choice == "Logistic Regression":
        log_reg_model = LogisticRegression()
        log_reg_model.fit(X_train, y_train)
        preds = log_reg_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, preds)

        st.write(f"### Logistic Regression Results")
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write(f"**AUC-ROC Score:** {roc:.2f}")

    elif model_choice == "Random Forest":
        rf_model = RandomForestClassifier(n_estimators=100)
        rf_model.fit(X_train, y_train)
        preds = rf_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, preds)

        st.write(f"### Random Forest Results")
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write(f"**AUC-ROC Score:** {roc:.2f}")

    elif model_choice == "CNN + LSTM":
        if image_dir:
            cnn_model = load_model("cnn_model.h5")  # Load pre-trained CNN
            lstm_model = load_model("lstm_model.h5")  # Load pre-trained LSTM
            
            st.write("CNN + LSTM Model Loaded ✅")

# SHAP Explainability for CNN
if model_choice == "CNN + LSTM":
    if image_dir and cnn_model:
        st.subheader("SHAP Explanation for CNN Model")

        # Define SHAP explainer
        global cnn_explainer
        if cnn_explainer is None:
            cnn_explainer = shap.Explainer(cnn_model, X_images[:50])
        
        # Compute SHAP values
        shap_values = cnn_explainer(X_images[:5])

        # Plot SHAP
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.image_plot(shap_values, X_images[:5])
        st.pyplot()
