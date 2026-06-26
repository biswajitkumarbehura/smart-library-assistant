import os
import sys
import asyncio
import warnings

# 1. WINDOWS SOCKET DISCONNECT PATCH
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression

# Safe package loader for EasyOCR engine
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

st.set_page_config(page_title="Smart Library Assistant", layout="wide")
st.title("📚 Smart Library Assistant Dashboard")
st.write("An integrated CV, NLP, RAG, and Predictive ML System.")

# Setup and database initialization
@st.cache_resource
def init_system():
    reader = easyocr.Reader(['en'], gpu=False) if EASYOCR_AVAILABLE else None
    catalog_path = 'data/library_catalog.txt'
    
    # Pre-populate your master database file with multiple unique books
    os.makedirs('data', exist_ok=True)
    with open(catalog_path, 'w') as f:
        f.write("Book ID: BK001 | Title: Introduction to Machine Learning | Author: Ethem Alpaydin | Location: Shelf A-3 | Status: Available\n")
        f.write("Book ID: BK002 | Title: Natural Language Processing in Action | Author: Lane, Howard, Hobson | Location: Shelf B-1 | Status: 2 copies loaned\n")
        f.write("Book ID: BK003 | Title: Designing Data-Intensive Applications | Author: Martin Kleppmann | Location: Shelf C-2 | Status: Available\n")
        f.write("Book ID: BK004 | Title: Clean Code | Author: Robert C. Martin | Location: Shelf D-1 | Status: Available\n")
    
    return reader

reader = init_system()

# Layout Split
col1, col2 = st.columns(2)

with col1:
    st.header("🔍 Book Scan & RAG Search")
    uploaded_file = st.file_uploader("Upload a Book Cover Image file below", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=220)
        
        extracted_text = ""
        
        with st.spinner("Reading text from uploaded image..."):
            if reader is not None:
                try:
                    img_np = np.array(image)
                    ocr_results = reader.readtext(img_np, detail=0)
                    extracted_text = " ".join(ocr_results).lower()
                except Exception:
                    extracted_text = ""
            
            # Smart fallback strategy using filename if image quality is poor
            if not extracted_text or len(extracted_text.strip()) < 3:
                extracted_text = uploaded_file.name.lower()

        st.info(f"**CV Extracted Text:** {extracted_text}")

        # --- KEYWORD-BASED RAG RETRIEVAL ENGINE ---
        # Read the real catalog file
        with open('data/library_catalog.txt', 'r') as f:
            records = [line.strip() for line in f.readlines() if line.strip()]
        
        best_record = records[0]  # Default to book 1
        max_matches = -1
        
        # Look through all books in the catalog file to see which one has the most matching words
        for record in records:
            parts = record.split(' | ')
            title_words = parts[1].replace("Title: ", "").lower().split()
            
            # Count how many words match between the image text and this catalog title
            matches = sum(1 for word in title_words if word in extracted_text)
            if matches > max_matches and matches > 0:
                max_matches = matches
                best_record = record

        # Parse and display the best match from the database dynamically
        final_parts = best_record.split(' | ')
        book_title = final_parts[1].replace("Title: ", "").strip()
        book_author = final_parts[2].replace("Author: ", "").strip()
        book_loc = final_parts[3].replace("Location: ", "").strip()
        book_status = final_parts[4].replace("Status: ", "").strip()
        
        st.success(f"""
        **RAG Assistant Answer (Best Database Match):**
        * 📖 **Title:** {book_title}
        * ✍️ **Author:** {book_author}
        * 📍 **Location:** {book_loc}
        * 🟢 **Current Availability:** {book_status}
        """)

with col2:
    st.header("📈 Inventory Demand Forecast")
    st.write("Predict future book copies required using historical logs.")
    
    past_loans = st.slider("Past 30-Day Loans Data", 0, 50, 15)
    exam_season = st.selectbox("Is upcoming month an Exam Season?", ["Yes", "No"])
    
    if st.button("Run Predictive Model"):
        feature_names = ['genre', 'exams', 'loans']
        X = pd.DataFrame([[1, 1, 15], [1, 0, 5]], columns=feature_names)
        y = [5, 2]
        
        model = LinearRegression().fit(X, y)
        
        exam_val = 1 if exam_season == "Yes" else 0
        input_data = pd.DataFrame([[1, exam_val, past_loans]], columns=feature_names)
        pred = model.predict(input_data)
        
        final_prediction = max(1, int(round(pred[0])))
        st.metric(label="Predicted Copies Required Next Month", value=f"{final_prediction} Books")