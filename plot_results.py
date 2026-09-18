import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def main(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 1 Feature reduction
    t = pd.read_csv(Path(args.results) / "table1_reduction.csv")
    stages = t["Stage"].tolist()[:3]
    features = t["Features"].tolist()[:3]
    plt.figure(figsize=(6.2,4.2))
    plt.plot(stages, features, marker="o")
    plt.xlabel("CDLF stage")
    plt.ylabel("Number of features")
    plt.title("Feature Reduction Curve")
    plt.grid(True, alpha=0.25)
    savefig(out / "fig2_feature_reduction.png")

    # 2 Sample reduction
    stages2 = ["Original", "CDLF"]
    samples = [t.loc[0, "Samples"], t.loc[3, "Samples"]]
    plt.figure(figsize=(6.2,4.2))
    plt.plot(stages2, samples, marker="o")
    plt.xlabel("Dataset representation")
    plt.ylabel("Training samples")
    plt.title("Sample Reduction Curve")
    plt.grid(True, alpha=0.25)
    savefig(out / "fig3_sample_reduction.png")

    # 3 Classification performance
    c = pd.read_csv(Path(args.results) / "table2_classification.csv")
    labels = c["Metric"].tolist()[:4]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(7,4.5))
    plt.bar(x-w/2, c["Baseline"].tolist()[:4], w, label="Baseline")
    plt.bar(x+w/2, c["CDLF"].tolist()[:4], w, label="CDLF")
    plt.xticks(x, labels)
    plt.ylabel("Percentage (%)")
    plt.title("Classification Performance")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(out / "fig4_classification_performance.png")

    # 4 Efficiency
    e = pd.read_csv(Path(args.results) / "table3_efficiency.csv")
    labels = ["Training time (min)", "RSS memory (GB)"]
    row1 = e[e["Measure"] == labels[0]].iloc[0]
    row2 = e[e["Measure"] == labels[1]].iloc[0]
    b = [row1["Baseline"], row2["Baseline"]]
    c2 = [row1["CDLF"], row2["CDLF"]]
    x = np.arange(2)
    plt.figure(figsize=(7,4.5))
    plt.bar(x-w/2, b, w, label="Baseline")
    plt.bar(x+w/2, c2, w, label="CDLF")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title("Computational Efficiency")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(out / "fig5_computational_efficiency.png")

    # 5 ROC
    p = np.load(Path(args.results) / "predictions.npz")
    from sklearn.metrics import roc_auc_score
    auc_b = roc_auc_score(p["y_test"], p["baseline_score"])
    auc_c = roc_auc_score(p["y_test"], p["cdlf_score"])
    plt.figure(figsize=(6.2,5))
    plt.plot(p["baseline_fpr"], p["baseline_tpr"], label=f"Baseline (AUC = {auc_b:.3f})")
    plt.plot(p["cdlf_fpr"], p["cdlf_tpr"], label=f"CDLF (AUC = {auc_c:.3f})")
    plt.plot([0,1],[0,1],"--",label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves of Baseline and CDLF")
    plt.legend()
    plt.grid(True, alpha=0.25)
    savefig(out / "fig7_roc_curves.png")

    # 6 Confusion matrix
    cm = p["cdlf_cm"]
    plt.figure(figsize=(5.5,4.8))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([0,1], ["Benign","Malware"])
    plt.yticks([0,1], ["Benign","Malware"])
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("CDLF Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i,j]:,}", ha="center", va="center")
    savefig(out / "fig8_cdlf_confusion_matrix.png")

    print("Figures written to:", out.resolve())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--outdir", default="outputs/actual/figures")
    main(p.parse_args())
