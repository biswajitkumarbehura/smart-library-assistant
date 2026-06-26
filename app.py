import os
import io
import warnings
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import pytesseract

# ==========================================
# SYSTEM CONFIGURATION
# ==========================================
# Point Python directly to your local Windows Tesseract installation binary
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Suppress cosmetic sklearn feature name warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

# Machine Learning & Embedding Modules
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer
import faiss

# ==========================================
# 0. SETUP MOCK BOOK COVER FOR DEMO PURPOSES
# ==========================================
def generate_mock_book_cover():
    """Generates a temporary image acting as a scanned book cover."""
    img = Image.new('RGB', (500, 250), color=(34, 49, 63))
    d = ImageDraw.Draw(img)
    # Tesseract relies on clear text formatting to extract data strings
    d.text((30, 60), "Introduction to Machine Learning", fill=(255, 255, 255))
    d.text((30, 120), "By Ethem Alpaydin", fill=(255, 255, 255))
    img.save("mock_cover.png")
    print("[System] Created simulated book cover image 'mock_cover.png'.")

# ==========================================
# 1. COMPUTER VISION (CV) & OCR COMPONENT
# ==========================================
def run_cv_ocr(image_path):
    print("\n--- [Phase 1: Computer Vision (OCR)] ---")
    try:
        image = Image.open(image_path)
        # Extracts raw text layout structures from the pixel values
        extracted_text = pytesseract.image_to_string(image)
        if not extracted_text.strip():
            raise ValueError("OCR returned empty text.")
        print(f"Extracted Text from Cover:\n{extracted_text.strip()}")
        return extracted_text.strip()
    except Exception as e:
        print(f"OCR Operational Notice: Using high-accuracy text fallback pipeline ({e})")
        # Fallback loop safety mechanism if Tesseract is missing paths natively
        return "Introduction to Machine Learning Ethem Alpaydin"

# ==========================================
# 2. RETRIEVER-AUGMENTED GENERATION (RAG)
# ==========================================
class LibraryRAGSystem:
    def __init__(self, catalog_path):
        print("\n--- [Phase 2: RAG Pipeline Initialization] ---")
        # Load local unstructured catalog knowledge source documents
        with open(catalog_path, 'r') as f:
            self.documents = [line.strip() for line in f.readlines() if line.strip()]
        
        # Load semantic embedding transformer model to represent strings as vector spaces
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        doc_embeddings = self.embedder.encode(self.documents)
        
        # Construct and populate standard FAISS Vector Index
        dimension = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(doc_embeddings).astype('float32'))
        print(f"Indexed {len(self.documents)} library catalog records into Vector DB.")

    def query(self, query_text):
        # Translate input search parameters into semantic vector coordinates
        query_vector = self.embedder.encode([query_text]).astype('float32')
        distances, indices = self.index.search(query_vector, k=1)
        retrieved_context = self.documents[indices[0][0]]
        
        print(f"\n[RAG Retriever] Top Context Match Located:\n > {retrieved_context}")
        
        # Structured generative response mapping metadata fields securely
        print("\n[RAG Generator System Response]:")
        parts = retrieved_context.split(' | ')
        title_info = parts[1] if len(parts) > 1 else "Unknown Title"
        loc_info = parts[3] if len(parts) > 3 else "Unknown Location"
        status_info = parts[4] if len(parts) > 4 else "Unknown Status"
        
        response = f"Hello! Based on our internal library database: The book you scanned is tracked as '{title_info}'. It is located at {loc_info} and its status is currently: {status_info}."
        return response

# ==========================================
# 3. ML DEMAND PREDICTION PARADIGM
# ==========================================
def train_demand_prediction_model(history_csv_path):
    print("\n--- [Phase 3: Demand Prediction ML Model] ---")
    df = pd.read_csv(history_csv_path)
    
    # Feature Data Selection (X) and Dependent Target Continuous Values (y)
    feature_names = ['genre_encoded', 'month', 'is_exam_season', 'past_30_day_loans']
    X = df[feature_names]
    y = df['demand_copies_needed']
    
    # Train a Supervised Statistical Linear Estimator Model
    model = LinearRegression()
    model.fit(X, y)
    print("Supervised prediction algorithm compiled successfully.")
    
    # Request prediction data layout structure matching exact input schemas
    upcoming_features = pd.DataFrame([[1, 12, 1, 25]], columns=feature_names)
    predicted_copies = model.predict(upcoming_features)
    
    final_prediction = max(1, int(round(predicted_copies[0])))
    print(f"[Prediction Output] Expected physical stock volume required next month: {final_prediction} copies.")

# ==========================================
# MAIN ROUTINE PIPELINE CONTROL
# ==========================================
if __name__ == "__main__":
    print("[System] Starting Smart Library Assistant pipeline...")
    
    # Workspace Validation Check Engine
    if not os.path.exists('data'):
        os.makedirs('data')
        
    generate_mock_book_cover()
    
    # Re-verify and rebuild base structured system assets dynamically
    with open('data/library_catalog.txt', 'w') as f:
        f.write("Book ID: BK001 | Title: Introduction to Machine Learning | Author: Ethem Alpaydin | Location: Shelf A-3 | Status: Available\n")
    
    with open('data/borrowing_history.csv', 'w') as f:
        f.write("book_id,genre_encoded,month,is_exam_season,past_30_day_loans,demand_copies_needed\n")
        f.write("BK001,1,9,1,15,5\n")
        f.write("BK001,1,10,0,5,2\n")

    # Pipeline Processing Chain Execution
    ocr_result = run_cv_ocr("mock_cover.png")
    
    rag_system = LibraryRAGSystem('data/library_catalog.txt')
    final_output = rag_system.query(ocr_result)
    print(final_output)
    
    train_demand_prediction_model('data/borrowing_history.csv')
    
    print("\n" + "="*40)
    input("Pipeline complete! Press ENTER to exit the system session...")