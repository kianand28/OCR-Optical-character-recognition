import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def display_image_with_boxes(img, boxes):
    fig, ax = plt.subplots(figsize=(12, 10))
    for (x, y, w, h) in boxes:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    st.pyplot(fig)

# Streamlit UI
st.title("OCR Text Extraction")
st.write("Upload an image of a passport to extract text using Tesseract OCR.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg", "pdf"])

if uploaded_file is not None:
  
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    st.image(img, caption="Uploaded Image", use_column_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    
    text1 = pytesseract.image_to_data(threshed, output_type=Output.DATAFRAME)
    text2 = pytesseract.image_to_string(threshed, lang="eng")
 
    text = text1[text1.conf != -1]

    lines = text.groupby('block_num')['text'].apply(list)
    conf = text.groupby(['block_num'])['conf'].mean()
 
    st.subheader("Extracted Text")
    st.write(text2)

    n_boxes = len(text1['text'])
    boxes = []
    for i in range(n_boxes):
        if int(text1['conf'][i]) > 60:
            (x, y, w, h) = (text1['left'][i], text1['top'][i], text1['width'][i], text1['height'][i])
            boxes.append((x, y, w, h))

    st.subheader("Image with Bounding Boxes for High Confidence Text")
    display_image_with_boxes(img, boxes)