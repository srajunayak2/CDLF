from pathlib import Path
import argparse
import pandas as pd

def latex_table(df, caption, label, percent_cols=None, digits=2):
    percent_cols = percent_cols or []
    x = df.copy()
    for c in x.columns:
        if c in percent_cols:
            x[c] = x[c].map(lambda v: f"{float(v):.{digits}f}\\%")
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\scriptsize\n"
        "\\begin{tabular}{|" + "c|"*len(x.columns) + "}\n\\hline\n"
        + " & ".join([f"\\textbf{{{c}}}" for c in x.columns]) + " \\\\\n\\hline\n"
        + "\n".join(" & ".join(map(str,row)) + " \\\\\n\\hline" for row in x.astype(str).values)
        + "\n\\end{tabular}\n\\end{table}\n"
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    r = Path(args.results)
    out = Path(args.out or (r/"latex_tables"))
    out.mkdir(parents=True, exist_ok=True)

    mapping = {
        "table1_reduction.csv": ("Feature and Sample Reduction","tab:feature_sample_reduction"),
        "table2_classification.csv": ("Classification Performance","tab:classification"),
        "table3_efficiency.csv": ("Computational Efficiency","tab:efficiency"),
        "table6_confusion_matrix.csv": ("Confusion Matrix for the CDLF Model","tab:confusion_matrix"),
    }

    for fn,(cap,lab) in mapping.items():
        f=r/fn
        if f.exists():
            df=pd.read_csv(f)
            tex=latex_table(df,cap,lab)
            (out/(fn.replace(".csv",".tex"))).write_text(tex,encoding="utf-8")

    print("LaTeX tables written to:",out.resolve())

if __name__=="__main__":
    main()
