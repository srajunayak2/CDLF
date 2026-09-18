
# Implementation checklist for the CDLF paper

## Required actual experiments

1. Prepare the actual EMBER feature matrix and labels.
2. Verify exact EMBER release/version.
3. Verify number of usable samples after cleaning.
4. Verify class distribution.
5. Verify original feature dimension.
6. Train/test split with stratification.
7. Fit MI on training data only.
8. Record selected MI feature count and MI ranking.
9. Fit PCA on training data only.
10. Record retained PCA dimensions and explained variance.
11. Run K-Means/MiniBatchKMeans in the PCA space.
12. Select the real observation nearest each centroid.
13. Train baseline RF on full training representation.
14. Train CDLF RF on compact representatives.
15. Evaluate both on the identical held-out test set.
16. Save predictions/probabilities.
17. Generate Accuracy, Precision, Recall, F1 and ROC-AUC.
18. Generate CDLF confusion matrix.
19. Generate ROC curves from actual predictions.
20. Measure training time under identical conditions.
21. Measure memory consistently; use the same definition for both models.
22. Run the ablation study.
23. Run repeated paired experiments for statistical validation.
24. Report mean difference, standard deviation, t statistic and p value.
25. If claiming formal equivalence, use a predefined equivalence margin and an appropriate equivalence test; a nonsignificant t-test alone is not proof of equivalence.
26. Generate final tables and figures from saved result files.
27. Replace all provisional values in the manuscript with measured values.

## Paper-specific items currently represented

The uploaded paper currently displays:
- 500 original features
- 180 after MI
- 50 after PCA
- 100,000 training samples
- 50,000 compact samples
- 90% feature reduction
- 50% sample reduction
- provisional baseline/CDLF performance values
- provisional computational values
- provisional statistical values
- a provisional confusion matrix

These should be treated as manuscript values until reproduced by the actual implementation.
