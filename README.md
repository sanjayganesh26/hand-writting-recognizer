# hand-writting-recognizer
# HANDWRITTEN-DIGIT-RECOGNIZATION
# 🧠 Handwritten Digit Recognition using CNN and Streamlit

This project demonstrates a deep learning-based handwritten digit recognition system built using **TensorFlow**, **Keras**, and **Streamlit**. The model is trained on the **Kaggle MNIST CSV dataset** and deployed as a web application where users can upload an image and receive instant predictions.

---

## 📌 Project Overview

- **Goal:** Classify handwritten digits (0–9) using a Convolutional Neural Network (CNN).
- **Input:** 28x28 grayscale PNG image of a handwritten digit.
- **Output:** Predicted digit shown in a user-friendly web app.
- **Interface:** Built using **Streamlit** for easy deployment and interaction.

---

## 📁 Folder Structure
├── app.py ← Streamlit application
├── digit_model.keras ← Trained CNN model
├── requirements.txt ← List of Python dependencies
├── mnist_train.csv ← (Optional) Kaggle MNIST training data
├── mnist_test.csv ← (Optional) Kaggle MNIST test data
└── README.md ← Project documentation (this file)


---

## 🚀 How It Works

1. Load and preprocess the dataset from CSV
2. Normalize, reshape, and one-hot encode the data
3. Train a CNN model on the MNIST dataset
4. Save the trained model as `digit_model.keras`
5. Build a Streamlit UI that:
   - Accepts a PNG image upload
   - Resizes and normalizes the image
   - Uses the model to predict the digit
   - Displays the predicted digit on the screen

---

## 🧪 Model Architecture (CNN)

- `Conv2D → MaxPooling2D → Conv2D → MaxPooling2D`
- `Flatten → Dense → Dropout → Output`
- Activation functions: ReLU and Softmax
- Optimizer: Adam
- Loss function: Categorical Crossentropy

---

## 🖥️ How to Run Locally

### 🧩 1. Install Python 3.10
TensorFlow works best with Python 3.10. Install it from:
[https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)

### 📦 2. Install Dependencies
Run this in terminal/command prompt:

```bash
pip install -r requirements.txt
```

### ✅ Example Summary for Report/Viva:
This project uses a CNN model trained on the MNIST dataset to recognize handwritten digits from 28x28 grayscale images. A working Python script and Streamlit interface are included, but only the input/output results are documented for simplicity. Model achieved over 98% accuracy.

### Google colab link
https://colab.research.google.com/drive/1yg-ZPlGhewpKLh316NCsfx2kOVjhHZiT?usp=sharing
