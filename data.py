import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. CREATE 1000 RECORDS WITH SOME NULL VALUES
# ------------------------------------------------------

np.random.seed(42)

categories = ["Electronics", "Soft Toys", "Clothes", "Books"]

data = {
    "product_id": range(1, 1001),
    "category": np.random.choice(categories, 1000),
    "price": np.random.randint(100, 2000, 1000).astype(float)
}

df = pd.DataFrame(data)

# Add null values randomly to price
null_indices = np.random.choice(df.index, 150, replace=False)
df.loc[null_indices, "price"] = np.nan

print("🔹 DATA CREATED WITH NULL VALUES")
print(df.head())

# ------------------------------------------------------
# 2. HANDLE MISSING VALUES USING ML MODEL
# ------------------------------------------------------

# We convert category → numeric
df["category_code"] = df["category"].astype("category").cat.codes

# TRAINING DATA (rows where price is NOT NULL)
train_data = df[df["price"].notnull()]
X_train = train_data[["category_code"]]
y_train = train_data["price"]

# MISSING DATA (rows where price IS NULL)
missing_data = df[df["price"].isnull()]
X_missing = missing_data[["category_code"]]

# Train simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict missing prices
predicted_prices = model.predict(X_missing)

# Fill missing prices
df.loc[df["price"].isnull(), "price"] = predicted_prices

print("\n🔹 MISSING VALUES FILLED SUCCESSFULLY")
print(df.head())

# ------------------------------------------------------
# 3. SAVE CLEANED DATA
# ------------------------------------------------------

df.to_csv("cleaned_data.csv", index=False)


# ------------------------------------------------------
# 4. PLOT GRAPH
# ------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(df["price"], bins=20)
plt.title("Price Distribution After Filling Missing Values")
plt.xlabel("Price")
plt.ylabel("Count")
plt.grid(True)
plt.show()

