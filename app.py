import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("my_model2.hdf5")
    return model

model_d = load_model()

# App title
st.title("Pneumonia Classification from Chest X-ray")

# File uploader widget
file = st.file_uploader("Please upload a chest X-ray image", type=["jpg", "png"])

# Function to preprocess and predict
def import_and_predict(image_data, model):
    size = (128, 128)  # Update to your model's input size if needed
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)
    image = image.convert("RGB") 
    img = np.asarray(image)
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)     # Add batch dimension
    prediction = model.predict(img)
    return prediction

# When a file is uploaded
if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction = import_and_predict(image, model_d)

    if prediction[0][0] > 0.5:
        st.markdown("### 🟥 Prediction: *Pneumonia Detected* 😷")
        st.markdown(f"*Confidence:* {prediction[0][0]*100:.2f}%")
    else:
        st.markdown("### 🟩 Prediction: *Normal* 😊")
        st.markdown(f"*Confidence:* {(1 - prediction[0][0])*100:.2f}%")