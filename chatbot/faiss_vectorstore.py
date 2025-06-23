# chatbot/faiss_vectorstore.py

import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

docs = []          # original text chunks
embeddings = None  # np.array of vectors
index = None       # FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text):
    return " ".join(text.split())

# STEP 1: Extract and embed documents
def load_and_index_pdfs(folder_path):
    global docs, embeddings, index
    docs.clear()

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            doc = fitz.open(pdf_path)
            text = ""

            for page in doc:
                text += page.get_text()

            chunks = text.split("\n\n")
            cleaned_chunks = [clean_text(chunk) for chunk in chunks if len(chunk.strip()) > 30]

            filtered_chunks = [
                chunk for chunk in cleaned_chunks
                if not any(bad in chunk.lower() for bad in ["table of", "contents", "index", "faq", "about us"])
            ]

            docs.extend(filtered_chunks)
            print(f"✅ Indexed {len(filtered_chunks)} chunks from {filename}")

    if not docs:
        return

    embeddings = model.encode(docs, show_progress_bar=True)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

# STEP 2: Search top-k similar chunks
def search_similar_chunks(query, top_k=3):
    if not index or not docs:
        return []

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)

    return [docs[i] for i in I[0] if i < len(docs)]
