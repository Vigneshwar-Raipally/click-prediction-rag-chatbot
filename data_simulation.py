import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Reproducibility
random.seed(42)
np.random.seed(42)

# Parameters
categories = ["Electronics", "Clothing", "Books", "Home Decor", "Toys"]
num_rows = 200
start_time = datetime.now()

data = []

# Generate simulated data with patterns RF can learn
for i in range(num_rows):
    timestamp = start_time + timedelta(seconds=i * random.randint(1, 3))
    user_id = 1000 + i
    category = random.choice(categories)
    time_on_page = round(np.random.uniform(30, 300), 2)
    price = round(np.random.uniform(100, 1000), 2)

    # Introduce a nonlinear pattern in click behavior
    # Example: Electronics and Toys users with high time and low price => more likely to click
    if category in ["Electronics", "Toys"]:
        clicked = 1 if (time_on_page > 180 and price < 500) else 0
    elif category == "Books":
        clicked = 1 if (price < 300 and time_on_page > 120) else 0
    else:
        clicked = np.random.choice([0, 1], p=[0.6, 0.4])  # Random-ish for others

    data.append([
        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        user_id,
        category,
        time_on_page,
        price,
        clicked
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "timestamp", "user_id", "product_category", "time_on_page", "price", "clicked"
])

# Save to CSV inside project folder
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "ecommerce_simulated_data.csv")
df.to_csv(output_path, index=False)

print(f"✅ Data saved at: {output_path}")
print(df.head())
