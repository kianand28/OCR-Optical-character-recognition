



import streamlit as st
import pytesseract
from PIL import Image
import re
import json
# from googletrans import Translator
import cv2
import os
from src.ocr_model import *

# Set the Tesseract command path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def translate_text(text, lang='es'):
    # Uncomment the following lines if Google Translator is configured
    # translator = Translator()
    # return translator.translate(text, dest=lang).text
    return text  # Temporarily return text without translation for testing

def validate_fields(extracted_data):
    # Regex patterns for validation
    dob_pattern = r"\b(\d{2}[-/]\d{2}[-/]\d{4})\b"
    name_pattern = r"\b([A-Za-z]+ [A-Za-z]+)\b"
    surname_pattern = r"Surname:\s*([A-Za-z]+)"
    given_name_pattern = r"Given Name:\s*([A-Za-z]+)"
    passport_no_pattern = r"Passport No:\s*(\w{8,9})"
    nationality_pattern = r"Nationality:\s*([A-Za-z]+)"

    dob_match = re.search(dob_pattern, extracted_data)
    name_match = re.search(name_pattern, extracted_data)
    surname_match = re.search(surname_pattern, extracted_data)
    given_name_match = re.search(given_name_pattern, extracted_data)
    passport_no_match = re.search(passport_no_pattern, extracted_data)
    nationality_match = re.search(nationality_pattern, extracted_data)
    
    # Return results in JSON format
    validation_result = {
        "DOB": dob_match.group(0) if dob_match else "Not found",
        "Name": name_match.group(0) if name_match else "Not found",
        "Surname": surname_match.group(1) if surname_match else "Not found",
        "Given Name": given_name_match.group(1) if given_name_match else "Not found",
        "Passport No.": passport_no_match.group(1) if passport_no_match else "Not found",
        "Nationality": nationality_match.group(1) if nationality_match else "Not found"
    }
    return validation_result  # Return dictionary directly

def main():
    st.title("KYC Verification OCR")
    
    # Upload document
    uploaded_file = st.file_uploader("Upload your document", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Passport Image", use_column_width=True)

        # Extract text from image
        extracted_text_result = extract_text("temp_image.png")
        print("Extracted Text:", extracted_text_result)  # Debugging line

        # Translate if necessary (currently returns the same text)
        translated_text = translate_text(extracted_text_result)

        # Validate fields
        validation_results = validate_fields(translated_text)

        # Display results in JSON format
        try:
            st.write("OCR Extraction:")
            st.json(validation_results)  # Display dictionary as JSON
        except Exception as e:
            st.error(f"Failed to display results as JSON: {e}")
            st.text(validation_results)  # Show the raw output for debugging

        # Status message
        st.success("Document verification completed.")

if __name__ == "__main__":
    main()




import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pytesseract import Output
import re

# Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to display image with bounding boxes
def display_image_with_boxes(img, boxes):
    fig, ax = plt.subplots(figsize=(12, 10))
    for (x, y, w, h) in boxes:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    st.pyplot(fig)

# Streamlit UI
st.title("OCR Text Extraction with Field Labels")
st.write("Upload an image of a passport to extract text with field labels.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg", "pdf"])

if uploaded_file is not None:
    # Load the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Display the original image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    
    # Extract text and data
    text1 = pytesseract.image_to_data(threshed, output_type=Output.DATAFRAME)
    raw_text = pytesseract.image_to_string(threshed, lang="eng")
    
    # Define regex patterns for fields
    patterns = {
        "Passport No.": r"([A-Z0-9]{9})",  # Assuming passport number is alphanumeric with length 9
        "Name": r"([A-Z]+(?:\s[A-Z]+)+)",  # Full name in uppercase
        "Date of Birth": r"(\d{2}/\d{2}/\d{4})",
        "Place of Issue": r"(?:Issued at|Place of Issue)\s([A-Za-z\s,]+)"
    }
    
    # Apply regex and extract fields
    labeled_data = {}
    for label, pattern in patterns.items():
        match = re.search(pattern, raw_text)
        labeled_data[label] = match.group(1) if match else "Not Found"
    
    # Display labeled data
    st.subheader("Extracted Information with Labels")
    for label, data in labeled_data.items():
        st.write(f"**{label}:** {data}")
    
    # Filter and display bounding boxes for high-confidence text
    n_boxes = len(text1['text'])
    boxes = []
    for i in range(n_boxes):
        if int(text1['conf'][i]) > 60:
            (x, y, w, h) = (text1['left'][i], text1['top'][i], text1['width'][i], text1['height'][i])
            boxes.append((x, y, w, h))
    
    st.subheader("Image with Bounding Boxes for High Confidence Text")
    display_image_with_boxes(img, boxes)
