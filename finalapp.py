import os
import torch
from transformers import AutoModel, AutoTokenizer
import streamlit as st
from torch import nn, optim
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    model_path = "c:\Users\asus\Desktop\chatbot\IIT ROORKEE\model.h5"  
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True)
    model.classifier = nn.Linear(model.config.hidden_size, 46)  
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

def preprocess_and_predict(image):
    image = image.convert("RGB")
    image = np.array(image)
    image_tensor = torch.tensor(image).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs

def search_keywords(outputs, keywords):
    matches = [keyword for keyword in keywords if keyword in str(outputs)]
    return matches

def main():
    st.title("OCR Model Training and Search Functionality")
    uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            outputs = preprocess_and_predict(image)
            st.write("Model Outputs:", outputs)

            keyword_input = st.text_input("Enter keywords to search (comma separated):", "")
            
            if st.button("Search Keywords"):
                if keyword_input:
                    keywords = [kw.strip() for kw in keyword_input.split(",")]
                    matches = search_keywords(outputs, keywords)

                    if matches:
                        st.success(f"Found Keywords: {', '.join(matches)}")
                    else:
                        st.warning("No keywords found in the outputs.")
                else:
                    st.warning("Please enter keywords to search.")

if __name__ == "__main__":
    main()
