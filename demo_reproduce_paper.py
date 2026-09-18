"""
Reproduces the numerical values currently displayed in the uploaded manuscript.

These values are explicitly treated as provisional/illustrative in the manuscript.
Use this script to reproduce the paper's current tables/graphs for formatting only,
not as a substitute for the actual EMBER experiment.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("outputs/demo")
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

# Table I
t1 = pd.DataFrame([
    ["Original",500,0,100000,0],
    ["After MI",180,64.0,100000,0],
    ["After PCA",50,90.0,100000,0],
    ["After Sampling",50,90.0,50000,50.0],
], columns=["Stage","Feat.","FR (%)","Samples","SR (%)"])
t1.to_csv(OUT/"tables"/"table1_feature_sample_reduction.csv", index=False)

# Table II
t2 = pd.DataFrame([
    ["Accuracy",98.20,97.90,0.30],
    ["Precision",98.00,97.80,0.20],
    ["Recall",98.40,98.10,0.30],
    ["F1-score",98.20,97.90,0.30],
    ["ROC-AUC",0.992,0.989,0.003],
], columns=["Metric","Baseline","CDLF","Difference"])
t2.to_csv(OUT/"tables"/"table2_classification.csv", index=False)

# Table III
t3 = pd.DataFrame([
    ["Training samples",100000,50000,50.0],
    ["Features",500,50,90.0],
    ["Training time (min)",180,105,41.7],
    ["Memory usage (GB)",12,7,41.7],
], columns=["Measure","Baseline","CDLF","Reduction (%)"])
t3.to_csv(OUT/"tables"/"table3_efficiency.csv", index=False)

# Table IV
t4 = pd.DataFrame([
    ["Accuracy",98.20,97.90,-0.30,0.66,-1.31,0.210,"NS"],
    ["Precision",98.00,97.80,-0.20,0.83,-1.08,0.294,"NS"],
    ["Recall",98.40,98.10,-0.30,0.76,-1.24,0.235,"NS"],
    ["F1-score",98.20,97.90,-0.30,0.72,-1.29,0.216,"NS"],
    ["ROC-AUC",99.20,98.90,-0.30,0.72,-1.17,0.258,"NS"],
], columns=["Metric","Baseline","CDLF","Mean Diff.","Std. Dev.","t","p","Decision"])
t4.to_csv(OUT/"tables"/"table4_statistical_validation.csv", index=False)

# Table V
t5 = pd.DataFrame([
    ["Baseline RF",98.20,98.20,180],
    ["MI + RF",98.05,98.04,150],
    ["PCA + RF",97.98,97.96,132],
    ["Sampling + RF",97.96,97.94,128],
    ["MI + PCA + RF",97.94,97.92,118],
    ["CDLF",97.90,97.90,105],
], columns=["Configuration","Accuracy (%)","F1-score (%)","Training Time (min)"])
t5.to_csv(OUT/"tables"/"table5_ablation.csv", index=False)

# Table VII
t7 = pd.DataFrame([
    ["CNN","EMBER","No","No",98.50,98.30,0.993],
    ["LSTM","EMBER","No","No",98.30,98.10,0.991],
    ["Random Forest","EMBER","No","No",98.10,98.00,0.990],
    ["PCA + SVM","EMBER","Yes","No",97.90,97.70,0.987],
    ["Proposed CDLF","EMBER","90%","50%",97.90,97.90,0.989],
], columns=["Method","Data","FR","SR","Acc. (%)","F1 (%)","AUC"])
t7.to_csv(OUT/"tables"/"table7_comparison.csv", index=False)

# Figures
plt.figure(figsize=(6.2,4.2))
plt.plot(["Original","After MI","After PCA"],[500,180,50],marker="o")
plt.xlabel("CDLF stage"); plt.ylabel("Number of features")
plt.title("Feature Reduction Curve"); plt.grid(True,alpha=.25)
plt.tight_layout(); plt.savefig(OUT/"figures"/"fig2_feature_reduction.png",dpi=300); plt.close()

plt.figure(figsize=(6.2,4.2))
plt.plot(["Original","CDLF"],[100000,50000],marker="o")
plt.xlabel("Dataset representation"); plt.ylabel("Training samples")
plt.title("Sample Reduction Curve"); plt.grid(True,alpha=.25)
plt.tight_layout(); plt.savefig(OUT/"figures"/"fig3_sample_reduction.png",dpi=300); plt.close()

labels=["Accuracy","Precision","Recall","F1-score"]
b=[98.2,98.0,98.4,98.2]; c=[97.9,97.8,98.1,97.9]
x=np.arange(4); w=.35
plt.figure(figsize=(7,4.5))
plt.bar(x-w/2,b,w,label="Baseline"); plt.bar(x+w/2,c,w,label="CDLF")
plt.xticks(x,labels); plt.ylabel("Percentage (%)"); plt.title("Classification Performance")
plt.legend(); plt.grid(axis="y",alpha=.25)
plt.tight_layout(); plt.savefig(OUT/"figures"/"fig4_classification_performance.png",dpi=300); plt.close()

plt.figure(figsize=(7,4.5))
plt.bar(np.arange(2)-w/2,[180,12],w,label="Baseline")
plt.bar(np.arange(2)+w/2,[105,7],w,label="CDLF")
plt.xticks(np.arange(2),["Training time (min)","Memory (GB)"])
plt.ylabel("Value"); plt.title("Computational Efficiency"); plt.legend()
plt.grid(axis="y",alpha=.25)
plt.tight_layout(); plt.savefig(OUT/"figures"/"fig5_computational_efficiency.png",dpi=300); plt.close()

ab_cfg=["Baseline","MI","PCA","Sampling","MI+PCA","CDLF"]
ab_acc=[98.20,98.05,97.98,97.96,97.94,97.90]
plt.figure(figsize=(6.5,4.2))
plt.plot(ab_cfg,ab_acc,marker="o")
plt.xlabel("Model configuration"); plt.ylabel("Accuracy (%)")
plt.title("Ablation Accuracy Curve"); plt.grid(True,alpha=.25)
plt.tight_layout(); plt.savefig(OUT/"figures"/"fig6_ablation_accuracy.png",dpi=300); plt.close()

# A schematic ROC matching the manuscript's reported AUC labels.
fpr=np.array([0,.001,.002,.005,.01,.02,.05,.1,.2,.4,1])
tpr_b=np.array([0,.55,.72,.82,.90,.94,.965,.98,.99,.995,1])
tpr_c=np.array([0,.52,.69,.80,.88,.93,.96,.975,.987,.994,1])
plt.figure(figsize=(6.2,5))
plt.plot(fpr,tpr_b,label="Baseline (AUC = 0.992)")
plt.plot(fpr,tpr_c,label="CDLF (AUC = 0.989)")
plt.plot([0,1],[0,1],"--",label="Random")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves of Baseline and CDLF Models")
plt.legend(); plt.grid(True,alpha=.25)
plt.tight_layout(); plt.savefig(OUT/"figures"/"fig7_roc_curves.png",dpi=300); plt.close()

cm=np.array([[9780,220],[200,9800]])
plt.figure(figsize=(5.5,4.8))
plt.imshow(cm,interpolation="nearest")
plt.xticks([0,1],["Benign","Malware"]); plt.yticks([0,1],["Benign","Malware"])
plt.xlabel("Predicted Class"); plt.ylabel("Actual Class")
plt.title("CDLF Confusion Matrix")
for i in range(2):
    for j in range(2):
        plt.text(j,i,f"{cm[i,j]:,}",ha="center",va="center")
plt.tight_layout(); plt.savefig(OUT/"figures"/"fig8_cdlf_confusion_matrix.png",dpi=300); plt.close()

print("Demo tables and figures created under:", OUT.resolve())
