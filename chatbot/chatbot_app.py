# chatbot/chatbot_app.py

import streamlit as st
from vectorstore import load_and_index_pdfs, search_docs
import os

st.set_page_config(page_title="📚 Ask Your PDFs (Offline)", layout="centered")
st.title("📚 Ask Your PDFs (Offline Chatbot Demo)")

# 🔍 Get correct absolute path
pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs")

# Index documents only once per session
if "indexed" not in st.session_state:
    with st.spinner("📂 Indexing PDFs..."):
        load_and_index_pdfs(pdf_dir)
        st.session_state.indexed = True
    st.success("✅ PDFs indexed successfully.")

# Ask a question
query = st.text_input("💬 Ask a question:")

# Show results
if query:
    with st.spinner("🤖 Searching your documents..."):
        results = search_docs(query)

        if results:
            st.success(f"✅ Found {len(results)} match(es):")
            for i, match in enumerate(results, 1):
                st.markdown(f"**🔹 Match {i}:**\n\n{match[:800]}...\n")
        else:
            st.warning("❌ No relevant content found.")
