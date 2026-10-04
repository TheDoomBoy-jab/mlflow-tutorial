import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set MLflow tracking server URI
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Iris_Classification")

# Load and split dataset
X, y = datasets.load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,

    "random_state": 42
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Calculate metrics
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

with mlflow.start_run(run_name="Logistic_Regression") as run:
    # Log parameters and metrics
    mlflow.log_params(params)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    # Infer model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train.head(),
        registered_model_name="tracking_quickstart"
    )

    # Load model back and make predictions
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(X_test)

    # Build results DataFrame
    result = X_test.copy()
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    print(result.head(4))