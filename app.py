import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"# Import necessary libraries
import streamlit as st
from PIL import Image
import pytesseract as pt


def perform_ocr(image):
   
    text = pt.image_to_string(image, lang="hin+eng")
    return text


def extract_keywords(text, keywords):
    keyword_matches = []
    for keyword in keywords:
        if keyword in text:
            keyword_matches.append(keyword)
    return keyword_matches


def main():
    st.title("OCR Text Extraction and Keyword Search")
    st.markdown("""
        <style>
            body {
                background-color: #f0f8ff; /* Light background color */
                color: #333; /* Dark text color */
            }
            h1 {
                text-align: center;
                color: #ff5733; /* Vibrant color for the title */
                font-size: 3em;
                font-family: 'Arial', sans-serif;
            }
            .header {
                text-align: center;
                color: #007BFF; /* Vibrant blue for subtitle */
                font-size: 1.5em;
            }
            .container {
                text-align: center;
                padding: 20px;
            }
            .stTextInput, .stButton {
                margin: 10px auto;
                width: 60%;
                border-radius: 10px;
                background-color: #007BFF; /* Blue background for input and button */
                color: white;
                font-size: 1.2em;
                border: none;
            }
            .stTextInput:focus, .stButton:focus {
                outline: none;
            }
            .result {
                background-color: #e6ffe6; /* Light green background for results */
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
                margin-top: 20px;
            }
            .stImage {
                margin: 20px auto;
            }
        </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image file for OCR processing", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

       
        extracted_text = perform_ocr(image)
        st.subheader("Extracted Text:")
        st.text_area("Extracted Text", value=extracted_text, height=200)

        keyword_input = st.text_input("Enter keywords to search (comma separated):", "")
        
        if st.button("Search Keywords"):
            if keyword_input:
                keywords = [kw.strip() for kw in keyword_input.split(",")]
                matches = extract_keywords(extracted_text, keywords)
                
                if matches:
                    st.success(f"Found Keywords: {', '.join(matches)}")
                else:
                    st.warning("No keywords found in the extracted text.")
            else:
                st.warning("Please enter keywords to search.")

if __name__ == "__main__":
    main()
