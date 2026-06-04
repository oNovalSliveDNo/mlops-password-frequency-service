from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import os
import yaml
import mlflow
from entropy import TextEntropyTransformer

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# init the model
if params["model"] == "linear":
    model = LinearRegression()
elif params["model"] == "rf":
    model = RandomForestRegressor()
elif params["model"] == "ridge":
    model = Ridge()


# Create the pipeline
pipeline_steps = [
    (
        "features",
        FeatureUnion(
            [
                # Vectorization feature
                (
                    "vect_tfidf",
                    Pipeline(
                        [
                            (
                                "vect",
                                CountVectorizer(
                                    ngram_range=(
                                        params["ngrams"]["min"],
                                        params["ngrams"]["max"],
                                    ),
                                    analyzer="char",
                                ),
                            ),
                            ("tfidf", TfidfTransformer())
                            if params["tfidf"]
                            else ("passthrough", "passthrough"),
                        ]
                    ),
                ),
                # Entropy feature
                ("entropy", TextEntropyTransformer()),
            ]
        ),
    ),
    # Classifier
    ("clf", model),
]

# Create the pipeline
pipeline = Pipeline(pipeline_steps)

# Fit the pipeline with your training data
train = pd.read_csv("data/train.csv")
pipeline.fit(train["Password"], train["Times"])

# Create a directory to store the model if it doesn't exist
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Save the pipeline to a binary file
joblib.dump(pipeline, f"{model_dir}/pipeline.joblib")

with mlflow.start_run() as run:
    mlflow.log_params(params)
    mlflow.sklearn.log_model(
        pipeline, artifact_path="model", input_example=train[["Password"]]
    )
