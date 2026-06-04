import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from dvclive import Live


# Function to calculate RMSLE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Load the pipeline
pipeline = joblib.load("models/pipeline.joblib")

# Read the test data
test = pd.read_csv("data/test.csv").dropna()

# Assuming 'text_column' is the column with text data and 'Times' is the target
X_test = test["Password"]
y_test = test["Times"]

# Make predictions
predictions = pipeline.predict(X_test)
print(f"Predictions: {predictions}")

# Calculate RMSE
rmse_score = rmse(y_test, predictions)
print(f"RMSE: {rmse_score}")

df = test.copy()
df["y_pred"] = predictions
df["error"] = (df["Times"] - df["y_pred"]).abs()
df["length"] = df["Password"].str.len()

with Live() as live:
    live.log_metric("RMSE", rmse_score)
    live.log_plot(
        "error", df.groupby("length").error.mean().reset_index(), x="length", y="error"
    )
    live.log_plot(
        "y_true vs y_pred (log)",
        df.sample(1000),
        x="Times",
        y="y_pred",
        template="scatter",
    )
    live.log_plot(
        "y_true vs y_pred",
        df.sample(1000).assign(y_pred=10**df.y_pred, Times=10**df.Times),
        x="Times",
        y="y_pred",
        template="scatter",
    )
