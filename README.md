# 📊 E-Commerce Real-Time Click Prediction & PDF Q&A Assistant  
### By Vigneshwar Raipally  
**Imarticus Data Science Internship - Assessment Cohort 5**

---

## 🚀 Project Overview

This project simulates an end-to-end data science solution:
- An ML-powered **real-time product click prediction dashboard**
- An **offline document Q&A chatbot** built using local PDF indexing
- A **final unified Streamlit app** combining both features in one place

---

## 🧠 Key Features

### 🔹 1. Real-Time Click Prediction
- Simulated user-product interaction data
- ML model (Logistic Regression) predicts click likelihood
- Dynamic dashboard with:
  - Category filtering
  - Real-time predictions
  - Visual insights (CTR, engagement)

### 🔹 2. RAG Chatbot (Retrieval-Augmented Generation)
- Uses local open-source PDFs (ML, Python, SQL, Streamlit)
- Text is indexed using **FAISS + Sentence Transformers**
- Responds to user questions with relevant document chunks

---

## 🗃️ Folder Structure

```
Final_Imarticus_Ecommerce_Project/
│
├── app.py                         # Standalone E-Commerce Prediction App
├── chatbot_app.py                # Basic keyword-matching chatbot (early version)
├── main_app.py                   # Final Streamlit app combining both features
├── data_simulation.py            # Generates simulated click data
├── train_model.py                # Trains the ML model
├── test_model.py                 # Evaluates model on test data
│
├── model/
│   ├── model.pkl                 # Trained model
│   ├── model_features.pkl        # Features used by model
│
├── chatbot/
│   ├── faiss_vectorstore.py     # Embedding + FAISS indexing
│   ├── rag_chatbot_app.py       # Vector-based retrieval chatbot using sentence-transformers
│   ├── vectorstore.py           # Early version of PDF loader with keyword search
│   ├── loader.py                # Helper for PDF extraction using PyMuPDF
│   └── pdfs/                    # Folder containing PDF documents
│       ├── streamlit_guide.pdf
│       ├── sql_cheatsheet.pdf
│       ├── ml_cheatsheet.pdf
│       ├── ...
│
├── ecommerce_simulated_data.csv
├── requirements.txt
└── README.md
```

---

## 🛠️ Step-by-Step Execution Flow

### 🧾 1. Generate Data
```bash
python data_simulation.py
```
This creates `ecommerce_simulated_data.csv`.

---

### 🧪 2. Train the Model
```bash
python model/train_model.py
```
- Trains **Logistic Regression, Decision Tree, Random Forest**
- Evaluates using classification report + confusion matrix
- Saves best model as `model/model.pkl`

---

### 🧪 3. Test the Model (Optional)
```bash
python model/test_model.py
```
- Loads test split and evaluates saved model

---

### 🌐 4. Run Dashboard App
```bash
streamlit run app.py
```
- Filters by category
- Predicts clicks
- Displays visualizations

---

### 📚 5. Test Chatbot App (Simple Keyword Version)
```bash
streamlit run chatbot/chatbot_app.py
```
- Uses basic keyword matching to fetch relevant PDF text
- Useful for simple use-cases (no ML embeddings)

---

### 💬 6. Test RAG-Based Chatbot (Advanced Search)
```bash
streamlit run chatbot/rag_chatbot_app.py
```
- Uses `sentence-transformers` and `FAISS`
- Matches question semantically with document chunks

---

### 🧩 7. Final Combined App
```bash
streamlit run main_app.py
```
- Combines both:
  - Click Prediction Dashboard (tab 1)
  - RAG Chatbot (tab 2)

---

📸 See the `screenshots/` folder for visual demo of all components.


## 🧪 Technologies Used

- **Python**, **Pandas**, **Scikit-learn**
- **Streamlit** for web UI
- **Joblib** for model saving
- **Matplotlib & Seaborn** for plots
- **FAISS + SentenceTransformers** for RAG
- **PyMuPDF (fitz)** for PDF parsing

---

## 📌 Notes

- If PDFs return irrelevant answers → Use simpler tutorials (not academic PDFs)
- No OpenAI or paid APIs were used – 100% offline embedding with FAISS
- `main_app.py` is the final product, but earlier apps (`app.py`, `chatbot_app.py`, etc.) are retained to show the learning journey.

---
