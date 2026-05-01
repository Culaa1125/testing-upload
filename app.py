import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Deteksi Paru-Paru", layout="centered")

st.title("Deteksi Penyakit Paru-Paru (X-ray)")
st.write("Upload gambar X-ray untuk mendapatkan prediksi.")

# =========================
# LOAD MODEL (DARI REPO)
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_parurasio801010.h5")
    return model

model = load_model()

# =========================
# LABEL KELAS (WAJIB SESUAI TRAINING)
# =========================
class_names = ["covid", "lung normal", "lung opacity", "viral pneumonia"]

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # =========================
    # PREPROCESSING
    # =========================
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PREDIKSI
    # =========================
    with st.spinner("Menganalisis gambar..."):
        prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # =========================
    # OUTPUT
    # =========================
    st.success(f"Hasil Prediksi: {class_names[predicted_class]}")
    st.info(f"Confidence: {confidence*100:.2f}%")

    # (opsional) grafik probabilitas
    st.subheader("Probabilitas Tiap Kelas")
    st.bar_chart(prediction[0])