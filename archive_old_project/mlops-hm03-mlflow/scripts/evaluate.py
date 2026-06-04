import pandas as pd
import json
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
import mlflow
from matplotlib import pyplot as plt


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

metrics = {"RMSE": rmse_score}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

# with mlflow.start_run() as run:
#     mlflow.log_metric("RMSE", rmse_score)


df = test.copy()
df["y_pred"] = predictions
df["error"] = (df["Times"] - df["y_pred"]).abs()
df["length"] = df["Password"].str.len()

# Start an MLflow run
with mlflow.start_run():
    mlflow.log_metric("RMSE", rmse_score)
    # Plot 1: error by length
    fig, ax = plt.subplots()
    df.groupby("length").error.mean().reset_index().plot(
        x="length", y="error", ax=ax, title="Error by Length"
    )
    mlflow.log_figure(fig, "error_by_length.png")
    plt.close(fig)

    # Plot 2: y_true vs y_pred (log) scatter plot
    fig, ax = plt.subplots()
    df.plot(
        kind="scatter",
        x="Times",
        y="y_pred",
        ax=ax,
        loglog=True,
        title="y_true vs y_pred (log)",
        s=1,
    )
    mlflow.log_figure(fig, "y_true_vs_y_pred_log.png")
    plt.close(fig)

    # Plot 3: y_true vs y_pred scatter plot
    fig, ax = plt.subplots()
    df_transformed = df.assign(
        y_pred=lambda x: 10**x.y_pred, Times=lambda x: 10**x.Times
    )
    df_transformed.plot(
        kind="scatter", x="Times", y="y_pred", ax=ax, title="y_true vs y_pred", s=1
    )
    mlflow.log_figure(fig, "y_true_vs_y_pred.png")
    plt.close(fig)

    mlflow.log_table(
        df.groupby("length").error.mean().reset_index(), "error_by_length.csv"
    )
