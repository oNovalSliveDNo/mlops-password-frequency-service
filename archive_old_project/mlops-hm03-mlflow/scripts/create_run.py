import mlflow

with mlflow.start_run() as run:
    print(run.info.run_id)
