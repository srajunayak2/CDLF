# Data preparation

The actual paper identifies EMBER as the intended benchmark.

The easiest interface for the CDLF code is an `.npz` file containing:

- `X`: numeric feature matrix of shape `(N, d)`
- `y`: binary labels of shape `(N,)`, where 0 = benign and 1 = malware.

Example:

```python
import numpy as np

X = ...  # shape (N, d)
y = ...  # shape (N,)

np.savez_compressed("data/ember_features.npz", X=X, y=y)
```

If your EMBER data is in CSV format, convert it to the same representation.

The manuscript currently states 500 original features and 100,000 training samples in Table I. Verify these values from the actual prepared dataset before final reporting.

## Recommended split

Use a stratified train/test split. The implementation fits MI, PCA and representative sampling using training data only.

## If you have the official EMBER feature files

If your EMBER release is already converted to NumPy/CSV, use the conversion script or directly create the `.npz` file. The package intentionally avoids silently assuming a particular EMBER release because feature dimensions and available files can differ between releases.
