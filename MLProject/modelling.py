from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_data():
    base_dir = Path(__file__).resolve().parent

    train_path = base_dir / "diabetes_preprocessing" / "train_processed.csv"
    test_path = base_dir / "diabetes_preprocessing" / "test_processed.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Outcome"])
    y_train = train_df["Outcome"]

    X_test = test_df.drop(columns=["Outcome"])
    y_test = test_df["Outcome"]

    return X_train, X_test, y_train, y_test


def main():
    base_dir = Path(__file__).resolve().parent

    mlflow.set_tracking_uri(f"file:{base_dir / 'mlruns'}")
    mlflow.set_experiment("Diabetes_CI_Experiment")

    X_train, X_test, y_train, y_test = load_data()

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_CI_Run"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)

        print("CI training selesai.")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")


if __name__ == "__main__":
    main()