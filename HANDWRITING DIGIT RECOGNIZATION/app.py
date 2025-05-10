import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained Keras model
model = load_model('digit_model.keras', compile=False)

# Streamlit page configuration
st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")

# App title and instructions
st.title("✍️ Handwritten Digit Recognition")
st.markdown("""
Upload a **28x28 grayscale PNG image** of a handwritten digit (0–9), and the model will predict what digit it is.
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your image (PNG only)", type=["png"])

if uploaded_file is not None:
    try:
        # Load and preprocess the image
        image = Image.open(uploaded_file).convert("L")  # Grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image_array = np.array(image).reshape(1, 28, 28, 1).astype("float32") / 255.0

        # Display uploaded image
        st.image(image, caption="Uploaded Image", width=150)

        # Predict using the model
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)

        # Show result
        st.success(f"🎯 Predicted Digit: **{predicted_digit}**")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {str(e)}")

else:
    st.info("📌 Please upload a **28x28 grayscale PNG** image.")
