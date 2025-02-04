import pandas as pd
import numpy as np
import joblib  # For saving/loading the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_model(df):
    df["GDP per capita"] = df["GDP"] / df["Population"]

    # Features and target
    X = df[['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']]
    y = df['Life Expectancy (IHME)']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained. MAE: {mae:.2f}")

    # Save the trained model
    joblib.dump(model, "random_forest_model.pkl")

    return model, model.feature_importances_

def load_model():
    """Loads the trained model from file."""
    return joblib.load("random_forest_model.pkl")

def predict_life_expectancy(model, gdp_per_capita, poverty_ratio, year):
    """Uses the trained model to predict Life Expectancy."""
    input_data = np.array([[gdp_per_capita, poverty_ratio, year]])
    return model.predict(input_data)[0]
