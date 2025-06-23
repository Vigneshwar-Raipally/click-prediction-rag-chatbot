# chatbot/vectorstore.py

import os
import fitz  # PyMuPDF
import re

# Global document store
docs = []

def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# STEP 1: Load and index PDF chunks
def load_and_index_pdfs(folder_path):
    global docs
    docs.clear()  # Reset before reloading

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            doc = fitz.open(pdf_path)
            text = ""

            for page in doc:
                text += page.get_text()

            chunks = text.split("\n\n")
            cleaned_chunks = [clean_text(chunk) for chunk in chunks if len(chunk.strip()) > 20]

            filtered_chunks = [
                chunk for chunk in cleaned_chunks
                if not any(skip in chunk.lower() for skip in ["table of", "contents", "faq", "index", "release", "copyright", "about us"])
            ]

            docs.extend(filtered_chunks)
            print(f"✅ Indexed {len(filtered_chunks)} chunks from {filename}")

    # Show preview
    for i, chunk in enumerate(docs[:2]):
        print(f"\n📘 Chunk {i+1}:\n{chunk[:300]}...\n")

# STEP 2: Simple keyword-based ranking
def search_docs(query, top_k=3):
    query = query.lower()
    results = []

    for chunk in docs:
        text = chunk.lower()
        if query in text:
            score = text.count(query)
            results.append((chunk, score))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in results[:top_k]]
