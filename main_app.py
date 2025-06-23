# main_app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# FAISS Chatbot
from chatbot.faiss_vectorstore import load_and_index_pdfs, search_similar_chunks

# ----- Page Config -----
st.set_page_config(page_title="🛍️ Click Prediction + PDF Chatbot", layout="wide")

# ----- Tab Layout -----
tab1, tab2 = st.tabs(["🛍️ Click Prediction Dashboard", "📚 Ask Your PDFs (Offline)"])

# ============================
# 📍 TAB 1: E-COMMERCE DASHBOARD
# ============================
with tab1:
    st.title("🛍️ Click Prediction Dashboard")
    st.markdown("Analyze product interactions and predict user clicks in real-time.")

    # Load model & features
    model = joblib.load("model/model.pkl")
    feature_columns = joblib.load("model/model_features.pkl")

    # Load data
    df = pd.read_csv("ecommerce_simulated_data.csv")

    # Sidebar filters
    st.sidebar.header("📊 Filter Options")
    categories = df["product_category"].unique().tolist()
    selected_categories = st.sidebar.multiselect("Choose Product Categories", options=categories, default=categories)

    # Filter
    df = df[df["product_category"].isin(selected_categories)]

    # Feature engineering
    df_fe = df.copy()
    le = LabelEncoder()
    df_fe["product_category_encoded"] = le.fit_transform(df_fe["product_category"])
    features = df_fe[["product_category_encoded", "time_on_page", "price"]]
    features = features.reindex(columns=feature_columns, fill_value=0)

    # Predict
    df["Predicted Click"] = model.predict(features)

    # KPIs
    st.subheader("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Total Products", len(df))
    col2.metric("🧠 Predicted Clicks", int(df["Predicted Click"].sum()))
    col3.metric("🎯 Click Rate", f"{df['Predicted Click'].mean() * 100:.2f}%")

    # Table
    st.subheader("📋 Predicted Interactions")
    st.dataframe(df[["timestamp", "user_id", "product_category", "time_on_page", "price", "Predicted Click"]])

    # Charts
    st.subheader("📈 Click Rate by Category")
    chart = df.groupby("product_category")["Predicted Click"].mean().sort_values()
    fig1, ax1 = plt.subplots()
    sns.barplot(x=chart.values, y=chart.index, palette="mako", ax=ax1)
    ax1.set_xlabel("Click Rate")
    ax1.set_ylabel("Product Category")
    st.pyplot(fig1)

    st.subheader("⏱️ Time Spent vs Click")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x="Predicted Click", y="time_on_page", palette="Set2", ax=ax2)
    ax2.set_title("Time on Page vs Predicted Click")
    st.pyplot(fig2)


# ============================
# 📍 TAB 2: PDF CHATBOT (Offline RAG)
# ============================
with tab2:
    st.title("📚 Ask Your PDFs (Offline Chatbot Demo)")
    st.markdown("Upload technical PDFs and ask questions — powered by FAISS + Sentence Transformers.")

    # Initialize and index PDFs once
    if "indexed" not in st.session_state:
        with st.spinner("🔄 Indexing PDFs..."):
            load_and_index_pdfs("chatbot/pdfs")
            st.session_state.indexed = True
        st.success("✅ Documents indexed successfully!")

    # User question
    query = st.text_input("💬 Enter your question about Python / ML / SQL:")

    # Answer area
    if query:
        with st.spinner("🤖 Searching for answers..."):
            results = search_similar_chunks(query)
            if results:
                st.success(f"🔍 Top {len(results)} Relevant Answer(s):")
                for i, text in enumerate(results, 1):
                    st.markdown(f"**🔹 Match {i}:**\n\n{text[:800]}...\n")
            else:
                st.warning("❌ No relevant content found.")
