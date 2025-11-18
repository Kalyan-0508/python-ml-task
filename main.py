# ===============================
#   COSMOS + FASTAPI + ML PROJECT
# ===============================

from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from bson import ObjectId

# Load .env
load_dotenv()

CONN_STRING = os.getenv("COSMOS_CONN_STRING")
DB_NAME = os.getenv("DB_NAME")
COLL_NAME = os.getenv("COLLECTION")

# FastAPI App
app = FastAPI()

# Connect to Cosmos
client = AsyncIOMotorClient(CONN_STRING, tls=True, tlsAllowInvalidCertificates=True)
db = client[DB_NAME]
collection = db[COLL_NAME]


# -------------------------
#   Helper – Fix Mongo ID
# -------------------------
def fix_id(doc):
    doc["id"] = str(doc["_id"])
    del doc["_id"]
    return doc


# -------------------------
#   Request Model
# -------------------------
class Product(BaseModel):
    name: str
    category: str
    price: float | None = None
    inStock: bool | None = True


# ======================================
#          API ROUTES
# ======================================

@app.get("/")
async def root():
    return {"message": "Cosmos + ML App Running"}


# -------------------------
#       CREATE PRODUCT
# -------------------------
@app.post("/products")
async def create_product(product: Product):
    data = product.dict()
    result = await collection.insert_one(data)
    return {"id": str(result.inserted_id)}


# -------------------------
#       READ ALL PRODUCTS
# -------------------------
@app.get("/products")
async def read_products():
    cursor = await collection.find().to_list(500)
    return [fix_id(p) for p in cursor]


# -------------------------
#       READ ONE PRODUCT
# -------------------------
@app.get("/products/{id}")
async def get_product(id: str):
    product = await collection.find_one({"_id": ObjectId(id)})
    if not product:
        raise HTTPException(404, "Product not found")
    return fix_id(product)


# -------------------------
#       UPDATE PRODUCT
# -------------------------
@app.put("/products/{id}")
async def update_item(id: str, data: dict):
    result = await collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": data}
    )
    if result.modified_count == 0:
        raise HTTPException(404, "Product not found")
    return {"message": "Updated"}


# -------------------------
#       DELETE PRODUCT
# -------------------------
@app.delete("/products/{id}")
async def delete_product(id: str):
    result = await collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 0:
        raise HTTPException(404, "Product not found")
    return {"message": "Deleted Successfully"}


# =================================================
#             MACHINE LEARNING IMPUTE LOGIC
# =================================================

@app.get("/ml/impute")
async def impute_missing_prices():
    """Reads data from DB → Finds missing prices → Predict using ML → Updates DB"""

    # Load products
    products = await collection.find().to_list(500)
    df = pd.DataFrame(products)

    if df.empty:
        raise HTTPException(400, "No data in database")

    # ML requires numeric category
    df["cat_code"] = df["category"].astype("category").cat.codes

    # Train on rows where price is available
    train_df = df[df["price"].notnull()]
    test_df = df[df["price"].isnull()]

    if train_df.empty:
        raise HTTPException(400, "Not enough data to train ML model")

    # Train model
    model = LinearRegression()
    model.fit(train_df[["cat_code"]], train_df["price"])

    # Predict missing values
    if not test_df.empty:
        predicted = model.predict(test_df[["cat_code"]])
        test_df["predicted_price"] = predicted

        # Update database for each missing row
        for (_, row), price in zip(test_df.iterrows(), predicted):
            await collection.update_one(
                {"_id": row["_id"]},
                {"$set": {"price": float(price)}}
            )

    return {"message": "Missing prices imputed successfully"}


# =================================================
#              ML PRICE PREDICTION
# =================================================

@app.post("/ml/predict")
async def predict_price(category: str):
    """Predict price for a new product"""
    
    products = await collection.find({"category": category}).to_list(500)
    df = pd.DataFrame(products)

    if df.empty:
        raise HTTPException(404, "No category data found")

    df["cat_code"] = df["category"].astype("category").cat.codes
    model = LinearRegression()

    model.fit(df[["cat_code"]], df["price"])
    cat_code = df["cat_code"].iloc[0]

    pred = model.predict([[cat_code]])

    return {"predicted_price": float(pred[0])}

