import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/generate-caption"

st.set_page_config(page_title="Image Caption Generator", layout="centered")

st.title("🖼️ Image Caption Generator")
st.write("Upload an image and get an AI-generated caption.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption 🚀"):
        with st.spinner("Generating caption..."):
            files = {"file": uploaded_image.getvalue()}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            caption = response.json()["caption"]
            st.success("Caption Generated!")
            st.markdown(f"### 📝 Caption:\n**{caption}**")
        else:
            st.error("Failed to generate caption")
