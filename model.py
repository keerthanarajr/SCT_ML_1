import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np

def train_model():
    # Load Kaggle dataset
    data = pd.read_csv("train.csv")

    # Create bathrooms feature (FullBath + HalfBath * 0.5)
    data["bathrooms"] = data["FullBath"] + 0.5 * data["HalfBath"]

    # Features and target
    X = data[["GrLivArea", "BedroomAbvGr", "bathrooms"]]
    y = data["SalePrice"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    
    # Save model
    os.makedirs("production_artifacts", exist_ok=True)
    joblib.dump(model, "production_artifacts/house_price_model.joblib")

    return model

def predict_price(area_sqft, bedrooms, bathrooms):
    model = joblib.load("production_artifacts/house_price_model.joblib")
    return model.predict([[area_sqft, bedrooms, bathrooms]])[0]

if __name__ == "__main__":
    train_model()