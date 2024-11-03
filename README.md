# OCR-Optical-character-recognition
# Text Extraction from Image

The project is a passport information extraction tool using tesseract-OCR v/s OpenAI LLM

# app.py - Text Extraction from Image using Tesseract-OCR 

This OCR Text Extraction uses Streamlit and Tesseract OCR to extract text from uploaded images of passports. The application allows users to upload an image file, processes it in grayscale, applies thresholding, and uses Tesseract to extract text with confidence scores. Text with high confidence is highlighted in bounding boxes and displayed on the image using matplotlib.

Key Points:
Text Extraction: Uses pytesseract to convert image text into a readable format.
Interactive UI: Streamlit interface for seamless image upload and display.
Bounding Box Visualization: High-confidence text areas are marked on the image.

Limitations of Tesseract OCR - Text overlaid designs, noisy backgrounds, or clutter can reduce OCR accuracy

To overcome on the limitations of Tesseract OCR, we implemented the project with OpenAI's LLM

# main.py - Text Extraction from Image using OpenAI's LLM

This processes images of passports to extract critical information (like passport number, issuing country, and expiration date) using OpenAI’s language model through the openai API. The project utilizes libraries such as base64 for image encoding, streamlit for creating the web interface, and langchain for chaining processes and parsing JSON output.

Key Points:
Automated Extraction: Uses OpenAI's LLM to analyze passport images and extract essential details in JSON format.
Modular Design: Implements dataclasses and pydantic for structured data management.
Streamlit Interface: Allows users to upload passport images and view extracted data in real-time.

