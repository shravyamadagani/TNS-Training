import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.title("🌍 Air Quality Index Prediction")

# Load Dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "16_air_quality_prediction.csv"))
# Preprocessing
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

df = df.dropna(subset=["predicted_aqi", "date"])

for col in ["pm2.5", "pm10", "temperature"]:
    df[col] = df[col].fillna(df[col].median())

df["location"] = df["location"].fillna(df["location"].mode()[0])

# Features and Target
X = df[["location", "pm2.5", "pm10", "temperature",
        "year", "month", "day"]]

y = df["predicted_aqi"]

X = pd.get_dummies(X, columns=["location"])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
st.write("MAE:", mean_absolute_error(y_test, y_pred))
st.write("R2 Score:", r2_score(y_test, y_pred))

# Model Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation Metrics")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("MAE", round(mae, 2))
col2.metric("MSE", round(mse, 2))
col3.metric("RMSE", round(rmse, 2))
col4.metric("R² Score", round(r2, 3))

# User Input
st.subheader("Enter Environmental Details")

location = st.selectbox("Location", df["location"].unique())
pm25 = st.number_input("PM2.5", min_value=0.0, value=25.0)
pm10 = st.number_input("PM10", min_value=0.0, value=50.0)
temperature = st.number_input("Temperature", value=30.0)
date = st.date_input("Date")

# Prediction
if st.button("Predict AQI"):

    data = pd.DataFrame({
        "location": [location],
        "pm2.5": [pm25],
        "pm10": [pm10],
        "temperature": [temperature],
        "year": [date.year],
        "month": [date.month],
        "day": [date.day]
    })

    data = pd.get_dummies(data, columns=["location"])
    data = data.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(data)[0]

    st.success(f"Predicted AQI: {prediction:.2f}")

    if prediction <= 50:
        st.info("Air Quality: Good")
    elif prediction <= 100:
        st.info("Air Quality: Satisfactory")
    elif prediction <= 200:
        st.warning("Air Quality: Moderate")
    elif prediction <= 300:
        st.warning("Air Quality: Poor")
    elif prediction <= 400:
        st.error("Air Quality: Very Poor")
    else:
        st.error("Air Quality: Severe")
