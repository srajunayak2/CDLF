
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from utils import load_dataset, clean_numeric

def representative_indices(Z,K,seed):
    km=MiniBatchKMeans(n_clusters=min(K,len(Z)),batch_size=2048,n_init=10,random_state=seed)
    lab=km.fit_predict(Z)
    reps=[]
    for c in range(km.n_clusters):
        ids=np.flatnonzero(lab==c)
        if len(ids):
            reps.append(ids[np.argmin(((Z[ids]-km.cluster_centers_[c])**2).sum(axis=1))])
    return np.asarray(reps)

def run_once(X,y,seed,a):
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=a.test_size,stratify=y,random_state=seed)

    base=RandomForestClassifier(n_estimators=a.n_estimators,n_jobs=a.n_jobs,random_state=seed).fit(Xtr,ytr)
    bp=base.predict(Xte); bs=base.predict_proba(Xte)[:,1]

    mi=mutual_info_classif(Xtr,ytr,random_state=seed)
    idx=np.argsort(mi)[::-1][:min(a.mi_features,Xtr.shape[1])]
    sc=StandardScaler().fit(Xtr[:,idx])
    tr=sc.transform(Xtr[:,idx]); te=sc.transform(Xte[:,idx])
    pca=PCA(n_components=min(a.pca_components,tr.shape[1]),random_state=seed).fit(tr)
    ztr=pca.transform(tr); zte=pca.transform(te)
    reps=representative_indices(ztr,a.clusters,seed)
    compact=RandomForestClassifier(n_estimators=a.n_estimators,n_jobs=a.n_jobs,random_state=seed).fit(ztr[reps],ytr[reps])
    cp=compact.predict(zte); cs=compact.predict_proba(zte)[:,1]

    def metrics(yt,p,s):
        return [
            accuracy_score(yt,p)*100,
            precision_score(yt,p,zero_division=0)*100,
            recall_score(yt,p,zero_division=0)*100,
            f1_score(yt,p,zero_division=0)*100,
            roc_auc_score(yt,s)*100
        ]
    return metrics(yte,bp,bs), metrics(yte,cp,cs)

def main(a):
    X,y=clean_numeric(*load_dataset(a.data))
    names=["Accuracy","Precision","Recall","F1-score","ROC-AUC"]
    base=[]; comp=[]
    for seed in range(a.start_seed,a.start_seed+a.runs):
        b,c=run_once(X,y,seed,a)
        base.append(b); comp.append(c)
        print("Completed seed",seed)
    base=np.asarray(base); comp=np.asarray(comp)

    rows=[]
    for j,n in enumerate(names):
        d=base[:,j]-comp[:,j]
        t,p=ttest_rel(base[:,j],comp[:,j])
        rows.append([n,base[:,j].mean(),comp[:,j].mean(),d.mean(),d.std(ddof=1),t,p,
                     "Significant" if p<=.05 else "NS"])
    df=pd.DataFrame(rows,columns=["Metric","Baseline","CDLF","Mean Diff.","Std. Dev.","t","p","Decision"])
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    pd.DataFrame(base,columns=["Accuracy","Precision","Recall","F1-score","ROC-AUC"]).assign(Model="Baseline",Seed=range(a.runs)).to_csv(out.with_name("repeated_baseline.csv"),index=False)
    pd.DataFrame(comp,columns=["Accuracy","Precision","Recall","F1-score","ROC-AUC"]).assign(Model="CDLF",Seed=range(a.runs)).to_csv(out.with_name("repeated_cdlf.csv"),index=False)
    print(df.to_string(index=False))
    print("Saved:",out.resolve())

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--mi-features",type=int,default=180)
    p.add_argument("--pca-components",type=int,default=50)
    p.add_argument("--clusters",type=int,default=50000)
    p.add_argument("--test-size",type=float,default=.20)
    p.add_argument("--start-seed",type=int,default=1)
    p.add_argument("--runs",type=int,default=10)
    p.add_argument("--n-estimators",type=int,default=300)
    p.add_argument("--n-jobs",type=int,default=-1)
    p.add_argument("--out",default="outputs/actual/table4_statistical_validation.csv")
    main(p.parse_args())
