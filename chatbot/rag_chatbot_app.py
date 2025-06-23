# chatbot/rag_chatbot_app.py

import streamlit as st
import os
from faiss_vectorstore import load_and_index_pdfs, search_similar_chunks

st.set_page_config(page_title="🧠 Semantic PDF Chatbot", layout="centered")
st.title("🧠 Ask Your PDFs (RAG-powered Chatbot)")

# 🔍 Get correct absolute path
pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs")

# Only index once per session
if "indexed" not in st.session_state:
    with st.spinner("📂 Embedding + Indexing PDFs..."):
        load_and_index_pdfs(pdf_dir)
        st.session_state.indexed = True
    st.success("✅ Semantic indexing complete.")

# Input
question = st.text_input("💬 Ask something:")

# Output
if question:
    with st.spinner("🔍 Searching using semantic similarity..."):
        matches = search_similar_chunks(question)

        if matches:
            st.success("✅ Top relevant context:")
            for i, chunk in enumerate(matches, 1):
                st.markdown(f"**🔹 Match {i}:**\n\n{chunk[:800]}...\n")
        else:
            st.error("❌ No relevant context found in the PDFs.")
