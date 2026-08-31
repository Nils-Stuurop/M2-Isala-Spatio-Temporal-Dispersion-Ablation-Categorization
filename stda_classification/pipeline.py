from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from scipy.io import loadmat
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SUPPORTED_SUPERVISED_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
}


def load_patient_dataset(path: str | Path) -> pd.DataFrame:
    """Load patient ablation data from CSV or MATLAB (.mat) files."""
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(dataset_path)

    if suffix == ".mat":
        mat_content = loadmat(dataset_path)
        table = {
            key: value.ravel()
            for key, value in mat_content.items()
            if not key.startswith("__") and hasattr(value, "ravel")
        }
        if not table:
            raise ValueError("No tabular variables found in MATLAB file")
        return pd.DataFrame(table)

    raise ValueError("Unsupported data format. Use CSV or MATLAB (.mat).")


def classify_patients_supervised(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = "recurrence",
    model: str = "random_forest",
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Train a supervised model and return recurrence prediction metrics."""
    if model not in SUPPORTED_SUPERVISED_MODELS:
        raise ValueError(f"Unsupported model '{model}'.")

    features = list(feature_columns)
    missing = [col for col in features + [target_column] if col not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    X = dataframe[features]
    y = dataframe[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = clone(SUPPORTED_SUPERVISED_MODELS[model])
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)

    metrics: dict[str, float | int] = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }

    if hasattr(classifier, "predict_proba") and pd.Series(y_test).nunique() > 1:
        y_score = classifier.predict_proba(X_test_scaled)[:, 1]
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
        except ValueError:
            pass

    return metrics


def cluster_patients_unsupervised(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign patients to unsupervised groups based on electrogram features."""
    features = list(feature_columns)
    missing = [col for col in features if col not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")
    if n_clusters > len(dataframe):
        raise ValueError("n_clusters cannot exceed the number of rows in dataframe.")

    X = dataframe[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    grouped = dataframe.copy()
    grouped["patient_group"] = clustering.fit_predict(X_scaled)
    return grouped


def quantify_recurrence_by_group(
    dataframe: pd.DataFrame,
    group_column: str = "patient_group",
    recurrence_column: str = "recurrence",
) -> pd.DataFrame:
    """Compute recurrence burden per derived patient group."""
    missing = [col for col in [group_column, recurrence_column] if col not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    summary = (
        dataframe.groupby(group_column, dropna=False)[recurrence_column]
        .agg(total_patients="count", recurrence_events="sum", recurrence_rate="mean")
        .reset_index()
        .sort_values(group_column)
    )
    summary["recurrence_rate"] = summary["recurrence_rate"].fillna(0.0)
    return summary
