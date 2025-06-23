import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Paths
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "ecommerce_simulated_data.csv")
model_path = os.path.join(project_dir, "model", "model.pkl")
features_path = os.path.join(project_dir, "model", "model_features.pkl")

# 2. Load dataset
df = pd.read_csv(data_path)
print(f"✅ Loaded dataset with {len(df)} rows.")

# 3. Encode category
le = LabelEncoder()
df['product_category_encoded'] = le.fit_transform(df['product_category'])

# 4. Define features and target
X = df[['product_category_encoded', 'time_on_page', 'price']]
y = df['clicked']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6. Load model and feature names
model = joblib.load(model_path)
model_name = model.__class__.__name__
feature_columns = joblib.load(features_path)
print(f"✅ Model loaded: {model_name}")

# 7. Check features
X_test = X_test.reindex(columns=feature_columns, fill_value=0)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate performance
print(f"\n📊 Performance on Test Data - {model_name}")
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"{model_name} - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
