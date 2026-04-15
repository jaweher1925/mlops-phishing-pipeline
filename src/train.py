import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

PROCESSED_PATH = params["data"]["processed_path"]
MAX_FEATURES = params["features"]["max_features"]
RANDOM_STATE = params["data"]["random_state"]

def load_data():
    train_df = pd.read_csv(f"{PROCESSED_PATH}/train.csv")
    test_df = pd.read_csv(f"{PROCESSED_PATH}/test.csv")
    return train_df, test_df

def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted")
    }

def train():
    train_df, test_df = load_data()

    X_train = train_df["text"].fillna("")
    y_train = train_df["label"]
    X_test = test_df["text"].fillna("")
    y_test = test_df["label"]

    # TF-IDF Vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Save vectorizer
    os.makedirs("models", exist_ok=True)
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Define 3 models
    models = {
        "logistic_regression": LogisticRegression(
            C=params["models"]["logistic_regression"]["C"],
            max_iter=params["models"]["logistic_regression"]["max_iter"],
            random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=params["models"]["random_forest"]["n_estimators"],
            max_depth=params["models"]["random_forest"]["max_depth"],
            random_state=RANDOM_STATE
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=50,
            random_state=RANDOM_STATE
        )
    }

    mlflow.set_experiment("phishing-email-classification")

    best_f1 = 0
    best_model_name = ""

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        with mlflow.start_run(run_name=model_name):

            # Log params
            mlflow.log_param("model", model_name)
            mlflow.log_param("max_features", MAX_FEATURES)
            mlflow.log_param("random_state", RANDOM_STATE)

            # Train
            model.fit(X_train_vec, y_train)

            # Evaluate
            y_pred = model.predict(X_test_vec)
            metrics = get_metrics(y_test, y_pred)

            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log model
            mlflow.sklearn.log_model(model, model_name)

            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")

            # Track best model
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model_name = model_name
                best_model = model

    # Save best model
    print(f"\nBest model: {best_model_name} with F1={best_f1:.4f}")
    with open("models/champion_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Save best model name
    with open("models/champion_name.txt", "w") as f:
        f.write(best_model_name)

    print("Champion model saved to models/champion_model.pkl")

if __name__ == "__main__":
    train()