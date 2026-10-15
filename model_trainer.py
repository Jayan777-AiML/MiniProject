import os
import json
import pickle

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

MODEL_DIR = "model"
METRICS_DIR = "metrics"

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METRICS_PATH = os.path.join(METRICS_DIR, "metrics.json")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")


def train_models():

    
    print("MODEL TRAINING")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training data loaded: {train_df.shape}")
    print(f"Testing data loaded:  {test_df.shape}")


    X_train = train_df.drop(columns=["Result"])
    y_train = train_df["Result"]

    X_test = test_df.drop(columns=["Result"])
    y_test = test_df["Result"]

    print(f"Number of features: {X_train.shape[1]}")


    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "GaussianNB": GaussianNB(),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
    }


    results = {}
    trained_models = {}

    for model_name, model in models.items():

        print(f"\nTraining {model_name}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[model_name] = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

        trained_models[model_name] = model

        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f1:.4f}")


    best_model_name = max(
        results,
        key=lambda name: results[name]["f1_score"]
    )

    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]

    print("                 BEST MODEL")
    print(f"Model     : {best_model_name}")
    print(f"Accuracy  : {best_metrics['accuracy']}")
    print(f"Precision : {best_metrics['precision']}")
    print(f"Recall    : {best_metrics['recall']}")
    print(f"F1 Score  : {best_metrics['f1_score']}")


    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


    with open(MODEL_PATH, "wb") as file:
        pickle.dump(best_model, file)

    print(f"\nModel saved: {MODEL_PATH}")


    metrics_data = {
        "best_model": best_model_name,
        "all_models": results
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics_data, file, indent=4)

    print(f"Metrics saved: {METRICS_PATH}")


    metadata = {
        "model_name": best_model_name,
        "model_version": "v1.0",
        "task": "Binary Classification",
        "dataset": "phishing.csv",
        "features": X_train.shape[1],
        "training_samples": X_train.shape[0],
        "testing_samples": X_test.shape[0],
        "target_column": "Result",
        "classes": {
            "0": "Legitimate",
            "1": "Phishing"
        }
    }

    with open(METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=4)

    print(f"Metadata saved: {METADATA_PATH}")

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    train_models()