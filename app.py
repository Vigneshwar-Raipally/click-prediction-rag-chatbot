import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Page Config ---
st.set_page_config(page_title="🧠 E-Commerce Click Prediction", layout="wide")

# --- Load Model + Data ---
project_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_dir, "model", "model.pkl")
features_path = os.path.join(project_dir, "model", "model_features.pkl")
data_path = os.path.join(project_dir, "ecommerce_simulated_data.csv")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)
df = pd.read_csv(data_path)

# --- UI Title ---
st.title("🛍️ E-Commerce Click Prediction Dashboard")
st.markdown("Use machine learning to predict if a user will click on a product based on their interaction data.")

# --- Sidebar Filters ---
st.sidebar.header("📊 Filter by Category")
categories = df['product_category'].unique().tolist()
selected_categories = st.sidebar.multiselect("Choose Categories", options=categories, default=categories)

# --- Filter Data ---
df = df[df['product_category'].isin(selected_categories)]

# --- Feature Engineering ---
df_fe = df.copy()
le = LabelEncoder()
df_fe['product_category_encoded'] = le.fit_transform(df_fe['product_category'])
features = df_fe[['product_category_encoded', 'time_on_page', 'price']]
features = features.reindex(columns=feature_columns, fill_value=0)

# --- Predict Clicks ---
df['Predicted Click'] = model.predict(features)

# --- Metrics ---
st.markdown("### 🔢 Dashboard Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("📦 Total Records", len(df))
col2.metric("🧠 Predicted Clicks", int(df['Predicted Click'].sum()))
col3.metric("🎯 CTR (%)", f"{df['Predicted Click'].mean() * 100:.2f}")

# --- Data Table ---
st.markdown("### 📋 Predicted Product Interactions")
st.dataframe(df[['timestamp', 'user_id', 'product_category', 'time_on_page', 'price', 'Predicted Click']], use_container_width=True)

# --- CTR by Category ---
st.markdown("### 📈 Click-Through Rate by Product Category")
ctr_by_cat = df.groupby("product_category")["Predicted Click"].mean().sort_values(ascending=False)
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.barplot(x=ctr_by_cat.values, y=ctr_by_cat.index, palette='coolwarm', ax=ax1)
ax1.set_xlabel("CTR")
ax1.set_title("Click Rate per Category")
st.pyplot(fig1)

# --- Time Spent vs Click Prediction ---
st.markdown("### ⏱️ Time on Page vs Predicted Click")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Predicted Click", y="time_on_page", palette="Set2", ax=ax2)
ax2.set_title("Time Spent on Product by Predicted Click")
ax2.set_xlabel("Predicted Click (0 = No, 1 = Yes)")
st.pyplot(fig2)
