from pymongo import MongoClient
import pandas as pd
import os

MONGO_URI = "mongodb+srv://jayan:jayan123@cluster0.3jv7qck.mongodb.net/?appName=Cluster0"
DATABASE_NAME = "db_sample"
COLLECTION_NAME = "sample_connection"

try:
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)

    # Select database and collection
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Fetch documents
    data = list(collection.find())

    print("Number of records:", len(data))

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Remove MongoDB _id
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    print("\nData shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Save CSV
    df.to_csv("data/phishing.csv", index=False)

    print("\nDataset saved successfully!")
    print("Location: data/phishing.csv")

except Exception as e:
    print("Error:", e)

finally:
    client.close()