import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve

from utils import load_dataset, clean_numeric, binary_metrics, save_json

def rss_gb():
    return psutil.Process().memory_info().rss / (1024 ** 3)

def select_mi_features(X_train, y_train, n_features, random_state):
    n_features = min(n_features, X_train.shape[1])
    mi = mutual_info_classif(
        X_train, y_train,
        discrete_features=False,
        random_state=random_state
    )
    idx = np.argsort(mi)[::-1][:n_features]
    return idx, mi

def select_representatives(Z, y, n_clusters, random_state):
    n_clusters = min(n_clusters, len(Z))
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=2048,
        n_init=10
    )
    labels = km.fit_predict(Z)
    centers = km.cluster_centers_

    # Select the nearest real observation to each centroid.
    representatives = []
    for c in range(n_clusters):
        members = np.flatnonzero(labels == c)
        if len(members) == 0:
            continue
        distances = np.sum((Z[members] - centers[c]) ** 2, axis=1)
        representatives.append(members[np.argmin(distances)])

    return np.array(representatives, dtype=int), km

def train_rf(X, y, random_state, n_estimators=300, n_jobs=-1):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=None
    ).fit(X, y)

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    metrics = binary_metrics(y_test, pred, score)
    cm = confusion_matrix(y_test, pred)
    fpr, tpr, thresholds = roc_curve(y_test, score)
    return metrics, cm, fpr, tpr, thresholds, pred, score

def main(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data)
    X, y = clean_numeric(X, y)

    if len(np.unique(y)) != 2:
        raise ValueError("Binary labels are required.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    # ---------------- BASELINE ----------------
    baseline_start = time.perf_counter()
    baseline = train_rf(
        X_train, y_train,
        args.random_state,
        args.n_estimators,
        args.n_jobs
    )
    baseline_time = time.perf_counter() - baseline_start
    baseline_memory = rss_gb()
    b_metrics, b_cm, b_fpr, b_tpr, b_thr, b_pred, b_score = evaluate(
        baseline, X_test, y_test
    )

    # ---------------- CDLF ----------------
    cdlf_start = time.perf_counter()

    mi_idx, mi_scores = select_mi_features(
        X_train, y_train, args.mi_features, args.random_state
    )
    Xtr_mi = X_train[:, mi_idx]
    Xte_mi = X_test[:, mi_idx]

    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr_mi)
    Xte_scaled = scaler.transform(Xte_mi)

    pca = PCA(
        n_components=min(args.pca_components, Xtr_scaled.shape[1]),
        random_state=args.random_state
    )
    Z_train = pca.fit_transform(Xtr_scaled)
    Z_test = pca.transform(Xte_scaled)

    reps, kmeans = select_representatives(
        Z_train, y_train, args.clusters, args.random_state
    )

    X_compact = Z_train[reps]
    y_compact = y_train[reps]

    compact_model = train_rf(
        X_compact, y_compact,
        args.random_state,
        args.n_estimators,
        args.n_jobs
    )
    cdlf_time = time.perf_counter() - cdlf_start
    cdlf_memory = rss_gb()

    c_metrics, c_cm, c_fpr, c_tpr, c_thr, c_pred, c_score = evaluate(
        compact_model, Z_test, y_test
    )

    # ---------------- SAVE ARRAYS ----------------
    np.savez_compressed(
        out / "predictions.npz",
        y_test=y_test,
        baseline_pred=b_pred,
        baseline_score=b_score,
        cdlf_pred=c_pred,
        cdlf_score=c_score,
        baseline_fpr=b_fpr,
        baseline_tpr=b_tpr,
        cdlf_fpr=c_fpr,
        cdlf_tpr=c_tpr,
        cdlf_cm=c_cm
    )

    np.save(out / "mi_scores.npy", mi_scores)
    np.save(out / "selected_feature_indices.npy", mi_idx)
    np.save(out / "representative_indices.npy", reps)

    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(pca, out / "pca.joblib")
    joblib.dump(kmeans, out / "kmeans.joblib")
    joblib.dump(compact_model, out / "cdlf_random_forest.joblib")
    joblib.dump(baseline, out / "baseline_random_forest.joblib")

    # ---------------- TABLE DATA ----------------
    original_d = X_train.shape[1]
    mi_d = len(mi_idx)
    pca_d = Z_train.shape[1]
    original_n = len(X_train)
    compact_n = len(reps)

    reduction = pd.DataFrame([
        ["Original", original_d, 0.0, original_n, 0.0],
        ["After MI", mi_d, (1-mi_d/original_d)*100, original_n, 0.0],
        ["After PCA", pca_d, (1-pca_d/original_d)*100, original_n, 0.0],
        ["After Sampling", pca_d, (1-pca_d/original_d)*100, compact_n,
         (1-compact_n/original_n)*100]
    ], columns=["Stage","Features","Feature_Reduction_pct","Samples","Sample_Reduction_pct"])
    reduction.to_csv(out / "table1_reduction.csv", index=False)

    class_df = pd.DataFrame([
        ["Accuracy", b_metrics["Accuracy"]*100, c_metrics["Accuracy"]*100],
        ["Precision", b_metrics["Precision"]*100, c_metrics["Precision"]*100],
        ["Recall", b_metrics["Recall"]*100, c_metrics["Recall"]*100],
        ["F1-score", b_metrics["F1-score"]*100, c_metrics["F1-score"]*100],
        ["ROC-AUC", b_metrics["ROC-AUC"], c_metrics["ROC-AUC"]],
    ], columns=["Metric","Baseline","CDLF"])
    class_df["Difference"] = class_df["Baseline"] - class_df["CDLF"]
    class_df.to_csv(out / "table2_classification.csv", index=False)

    efficiency = pd.DataFrame([
        ["Training samples", original_n, compact_n, (1-compact_n/original_n)*100],
        ["Features", original_d, pca_d, (1-pca_d/original_d)*100],
        ["Training time (min)", baseline_time/60, cdlf_time/60,
         (1-cdlf_time/baseline_time)*100 if baseline_time else np.nan],
        ["RSS memory (GB)", baseline_memory, cdlf_memory,
         (1-cdlf_memory/baseline_memory)*100 if baseline_memory else np.nan]
    ], columns=["Measure","Baseline","CDLF","Reduction_pct"])
    efficiency.to_csv(out / "table3_efficiency.csv", index=False)

    pd.DataFrame(
        b_cm, index=["Actual Benign","Actual Malware"],
        columns=["Predicted Benign","Predicted Malware"]
    ).to_csv(out / "table6_confusion_matrix.csv")

    meta = {
        "dataset": str(args.data),
        "samples_after_cleaning": int(len(X)),
        "training_samples": int(original_n),
        "test_samples": int(len(X_test)),
        "original_features": int(original_d),
        "mi_features": int(mi_d),
        "pca_components": int(pca_d),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "compact_samples": int(compact_n),
        "random_state": int(args.random_state),
        "test_size": float(args.test_size),
        "n_estimators": int(args.n_estimators),
        "clusters": int(args.clusters),
    }
    save_json(meta, out / "experiment_metadata.json")

    pd.DataFrame({
        "Metric": list(b_metrics.keys()),
        "Baseline": list(b_metrics.values()),
        "CDLF": list(c_metrics.values())
    }).to_csv(out / "metrics_raw.csv", index=False)

    print("\nCDLF experiment completed.")
    print(json.dumps(meta, indent=2))
    print("\nBaseline:", b_metrics)
    print("CDLF:", c_metrics)
    print("\nOutput directory:", out.resolve())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--mi-features", type=int, default=180)
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--clusters", type=int, default=50000)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--outdir", default="outputs/actual")
    main(p.parse_args())
