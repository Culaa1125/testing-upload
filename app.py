import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

st.write("App started")

st.set_page_config(page_title="Deteksi Paru-Paru", layout="centered")
st.title("Deteksi Penyakit Paru-Paru (X-ray)")
st.write("Upload gambar X-ray untuk mendapatkan prediksi.")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    import os
    st.write("File exists:", os.path.exists("model_parurasio801010.onnx"))
    st.write("Files in directory:", os.listdir("."))
    session = ort.InferenceSession("model_parurasio801010.onnx")
    return session

model = load_model()

# =========================
# LABEL KELAS
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
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # ONNX butuh float32

    # =========================
    # PREDIKSI
    # =========================
    with st.spinner("Menganalisis gambar..."):
        input_name = model.get_inputs()[0].name
        prediction = model.run(None, {input_name: img_array})[0]

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # =========================
    # OUTPUT
    # =========================
    st.success(f"Hasil Prediksi: {class_names[predicted_class]}")
    st.info(f"Confidence: {confidence*100:.2f}%")

    st.subheader("Probabilitas Tiap Kelas")
    st.bar_chart(prediction[0])
