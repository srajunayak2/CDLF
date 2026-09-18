from pathlib import Path
import json
import time
import tracemalloc
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

def load_dataset(path):
    """Load X/y from NPZ, NPY pair directory, CSV, or Parquet."""
    path = Path(path)

    if path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=False)
        if "X" not in z or "y" not in z:
            raise ValueError("NPZ must contain arrays named X and y.")
        return z["X"], z["y"]

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        label_candidates = ["y", "label", "target", "class"]
        label_col = next((c for c in label_candidates if c in df.columns), None)
        if label_col is None:
            raise ValueError("CSV must contain one of: y, label, target, class")
        y = df[label_col].to_numpy()
        X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).to_numpy()
        return X, y

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        label_candidates = ["y", "label", "target", "class"]
        label_col = next((c for c in label_candidates if c in df.columns), None)
        if label_col is None:
            raise ValueError("Parquet must contain one of: y, label, target, class")
        y = df[label_col].to_numpy()
        X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).to_numpy()
        return X, y

    raise ValueError(f"Unsupported data format: {path}")

def clean_numeric(X, y):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)

    finite = np.isfinite(X).all(axis=1)
    X, y = X[finite], y[finite]

    # Remove exact duplicate feature rows.
    _, unique_idx = np.unique(X, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    X, y = X[unique_idx], y[unique_idx]

    return X, y

def binary_metrics(y_true, y_pred, y_score):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }
    return out

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def timed_memory_call(func, *args, **kwargs):
    """Return result, elapsed seconds and Python peak traced memory."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 ** 3)

def matrix_to_dataframe(cm):
    return pd.DataFrame(
        cm,
        index=["Benign", "Malware"],
        columns=["Benign", "Malware"]
    )
