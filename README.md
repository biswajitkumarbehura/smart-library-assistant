# 📚 Smart Library Assistant Dashboard

An integrated, end-to-end AI prototype that combines **Computer Vision (CV)**, **Retrieval-Augmented Generation (RAG)**, and **Predictive Machine Learning** into a single dashboard. This system automates book inventory tracking via cover scans and forecasts upcoming resource demands using historical logs.

---

## 🛠️ Tech Stack & Key Architectures

* **Frontend Dashboard:** Streamlit
* **Computer Vision (OCR):** EasyOCR / PyTesseract
* **Embeddings & NLP:** Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Vector Database Engine:** FAISS (Facebook AI Similarity Search)
* **Predictive Analytics:** Scikit-Learn (Linear Regression)

---

## 🏗️ System Workflow

The system is split into two core operating modules:

### 1. Book Scan & RAG Search (CV + NLP)
* **Image Processing:** Users upload a book cover image. The system processes the image using an OCR engine to read text phrases directly from the cover layout.
* **Semantic Vector Search:** The extracted text is converted into a high-dimensional vector embedding. The system uses a local FAISS index to find the most contextually similar book record inside `data/library_catalog.txt`.
* **Structured Output:** It parses and displays specific metadata fields (Title, Author, Shelf Location, and Availability Status) clearly without internet hallucinations.

### 2. Inventory Demand Forecast (Supervised ML)
* **Regression Modeling:** A supervised machine learning algorithm evaluates historical borrowing logs.
* **Dynamic Scaling:** By adjusting variables such as past 30-day loan counts and upcoming exam schedules, librarians receive real-time, rounded recommendations indicating how many physical copies of a specific asset need to be stocked for the upcoming month.

---

## 🚀 How to Run the Project Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/biswajitkumarbehura/smart-library-assistant.git](https://github.com/biswajitkumarbehura/smart-library-assistant.git)
cd smart-library-assistant