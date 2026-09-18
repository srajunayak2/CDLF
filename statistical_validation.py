import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

def paired_table(baseline, compact, metric_names):
    rows = []
    for i, name in enumerate(metric_names):
        a = np.asarray(baseline[i], dtype=float)
        c = np.asarray(compact[i], dtype=float)
        if len(a) != len(c):
            raise ValueError("Paired arrays must have equal length.")
        diff = a - c
        t, p = ttest_rel(a, c)
        rows.append({
            "Metric": name,
            "Baseline_Mean": a.mean(),
            "CDLF_Mean": c.mean(),
            "Mean_Diff": diff.mean(),
            "Std_Dev": diff.std(ddof=1),
            "t": t,
            "p": p,
            "Decision": "Significant" if p <= 0.05 else "NS"
        })
    return pd.DataFrame(rows)

def main(args):
    df = pd.read_csv(args.input)
    metric_names = df["Metric"].tolist()
    baseline_cols = [c for c in df.columns if c.startswith("Baseline_")]
    cdlf_cols = [c for c in df.columns if c.startswith("CDLF_")]

    if len(baseline_cols) != 1 or len(cdlf_cols) != 1:
        raise ValueError(
            "Input must have columns Metric, Baseline_values and CDLF_values "
            "where each values column contains paired observations."
        )

    # Values are semicolon-separated within each row.
    rows = []
    for _, r in df.iterrows():
        b = np.array([float(x) for x in str(r[baseline_cols[0]]).split(";")])
        c = np.array([float(x) for x in str(r[cdlf_cols[0]]).split(";")])
        if len(b) != len(c):
            raise ValueError(f"Unequal paired observations for {r['Metric']}")
        t, p = ttest_rel(b, c)
        d = b-c
        rows.append([
            r["Metric"], b.mean(), c.mean(), d.mean(),
            d.std(ddof=1), t, p, "Significant" if p <= 0.05 else "NS"
        ])

    out = pd.DataFrame(rows, columns=[
        "Metric","Baseline","CDLF","Mean Diff.","Std. Dev.","t","p","Decision"
    ])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
