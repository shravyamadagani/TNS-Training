# Step 1: Import all necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Read the dataset
df = pd.read_csv("16_air_quality_prediction.csv")

# Step 3: Explore the dataset
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

# Step 4: Data preprocessing
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# Handle missing values
df = df.dropna(subset=["predicted_aqi", "date"]).copy()

for col in ["pm2.5", "pm10", "temperature"]:
    df[col] = df[col].fillna(df[col].median())

df["location"] = df["location"].fillna(
    df["location"].mode()[0]
)

# Step 5: Split features and target
X = df[
    [
        "location",
        "pm2.5",
        "pm10",
        "temperature",
        "year",
        "month",
        "day"
    ]
]

y = df["predicted_aqi"]

# Convert categorical location into numerical columns
X = pd.get_dummies(X, columns=["location"])

# Step 6: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Step 7: Create and train Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Step 8: Predict AQI
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Regression Results")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 3))

# Display actual and predicted values
results = pd.DataFrame({
    "Actual AQI": y_test.values,
    "Predicted AQI": y_pred
})

print("\nActual vs Predicted AQI:")
print(results.head(10).round(2))

# Step 10: Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.sort_values().plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Step 11: Actual vs Predicted graph
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.tight_layout()
plt.show()