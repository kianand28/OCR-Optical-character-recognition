import streamlit as st
import pytesseract
from PIL import Image
import cv2
from transformers import pipeline
import numpy as np
from transformers import T5Tokenizer

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
custom_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tesseract.exe"'

# Initialize translation model
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")

# Streamlit App
st.title("Text extraction and Translation for German product")

st.sidebar.title("Options")
st.sidebar.markdown(
    "Upload an image containing German text, extract the text, and translate it into English."
)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file) 
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Extract Text
    st.subheader("Extracted Text")
    extracted_text = pytesseract.image_to_string(gray, lang="deu", config=custom_config)

    if extracted_text:
        st.text_area("Detected German Text:", extracted_text, height=200)

        # Translate Text
        st.subheader("Translated Text")
        translated_text = translator(extracted_text)
        translation = translated_text[0]['translation_text']
        st.text_area("Translated English Text:", translation, height=200)

        # Provide Download Options
        st.subheader("Download Options")
        st.download_button("Download Extracted Text", extracted_text)
        st.download_button("Download Translated Text", translation)
    else:
        st.warning("No text detected. Try uploading a clearer image.")
else:
    st.info("Awaiting image upload.")

st.sidebar.markdown("## About")
st.sidebar.info(
    "This application uses Tesseract OCR for text extraction and Hugging Face's MarianMT for translation."
)
