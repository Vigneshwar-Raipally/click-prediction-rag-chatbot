import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Load the data
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "ecommerce_simulated_data.csv")
df = pd.read_csv(data_path)
print(f"✅ Dataset Loaded. Total rows: {len(df)}")

# 2. Encode categorical column
le = LabelEncoder()
df['product_category_encoded'] = le.fit_transform(df['product_category'])

# 3. Define features and target
X = df[['product_category_encoded', 'time_on_page', 'price']]
y = df['clicked']

# 4. Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Define models to compare
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 6. Evaluate all models
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    print(f"\n🔍 Evaluating: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Select best model
    if acc >= best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# 7. Save best model and features
model_dir = os.path.join(project_dir, "model")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(model_dir, "model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(model_dir, "model_features.pkl"))

print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.2f})")
print("💾 Saved as model.pkl and model_features.pkl")
