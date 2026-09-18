# Compact Data Learning Framework (CDLF) — Implementation Package

This package is prepared from the methodology and tables/figures in the uploaded paper:

**A Compact Data Learning Framework for Malware Detection with Joint Feature and Sample Reduction Under Statistical Equivalence Constraints**

The paper defines the pipeline:

EMBER → preprocessing → Mutual Information (MI) feature selection → PCA compression → K-Means representative sample selection → Random Forest → evaluation/statistical validation.

## Important reproducibility note

The PDF contains several **provisional/illustrative result values**. In particular, Tables I–V contain numerical values, while the manuscript itself states that final experimental values should come from the actual implementation. Therefore:

- `demo_reproduce_paper.py` reproduces the *paper's displayed/provisional numbers and graphs* for formatting/checking purposes.
- `run_cdlf.py` performs the **actual CDLF experiment** on supplied data and generates measured results.
- Do not report demo values as measured experimental evidence unless the actual experiment reproduces them.

## Package contents

- `src/run_cdlf.py` — complete CDLF training/evaluation pipeline.
- `src/demo_reproduce_paper.py` — reproduces the current manuscript tables/figures from the displayed values.
- `src/make_tables.py` — creates CSV and LaTeX versions of Tables I–VII from result files.
- `src/statistical_validation.py` — paired statistical analysis.
- `src/plot_results.py` — creates publication-ready feature/sample, classification, efficiency, ablation, ROC and confusion-matrix figures.
- `src/utils.py` — common data loading and metric utilities.
- `requirements.txt` — Python dependencies.
- `overleaf_tables.tex` — ready-to-copy LaTeX table definitions.
- `data/README_DATA.md` — instructions for preparing the actual dataset.

## Quick demo

```bash
python src/demo_reproduce_paper.py
```

This creates paper-style figures under `outputs/demo/` and CSV/LaTeX tables under `outputs/demo/tables/`.

## Actual experiment

1. Put the prepared EMBER feature matrix and labels into `data/`.
2. See `data/README_DATA.md`.
3. Example:

```bash
python src/run_cdlf.py \
  --data data/ember_features.npz \
  --mi-features 180 \
  --pca-components 50 \
  --clusters 50000 \
  --test-size 0.20 \
  --random-state 42 \
  --outdir outputs/actual
```

The `--clusters` value should be chosen so that the number of selected representatives matches the intended sample reduction. It must not exceed the number of training samples.

For the manuscript's current displayed configuration:

- Original features: 500
- After MI: 180
- After PCA: 50
- Original training samples: 100,000
- Compact training samples: 50,000

These are manuscript values, not automatically guaranteed to be the actual EMBER dimensions/counts in your experiment.

## Outputs from the actual pipeline

The actual pipeline saves:

- baseline and CDLF predictions
- accuracy, precision, recall, F1 and ROC-AUC
- confusion matrix
- ROC curve
- training time
- peak memory where available
- feature/sample reduction
- ablation results
- paired statistical validation
- CSV tables
- LaTeX tables
- PNG figures

## Leakage control

MI fitting, PCA fitting and K-Means representative selection are performed on the training data only. The held-out test set remains unchanged and is used only for final evaluation.

## Recommended final-paper workflow

Run the actual experiment first. Then use the generated CSV files to populate the manuscript. Do not manually type provisional values into the final paper.
