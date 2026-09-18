
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from utils import load_dataset, clean_numeric

def reps_from_kmeans(Z, n_clusters, random_state):
    km = MiniBatchKMeans(
        n_clusters=min(n_clusters, len(Z)),
        random_state=random_state,
        batch_size=2048,
        n_init=10
    )
    labels = km.fit_predict(Z)
    reps=[]
    for c in range(km.n_clusters):
        idx=np.flatnonzero(labels==c)
        if len(idx):
            dist=((Z[idx]-km.cluster_centers_[c])**2).sum(axis=1)
            reps.append(idx[np.argmin(dist)])
    return np.asarray(reps)

def rf(X,y,seed,n_estimators,n_jobs):
    return RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs
    ).fit(X,y)

def evaluate(model,X,y):
    p=model.predict(X)
    return accuracy_score(y,p)*100, f1_score(y,p,zero_division=0)*100

def main(a):
    X,y=clean_numeric(*load_dataset(a.data))
    Xtr,Xte,ytr,yte=train_test_split(
        X,y,test_size=a.test_size,stratify=y,random_state=a.random_state
    )

    rows=[]

    # Baseline
    t=time.perf_counter()
    m=rf(Xtr,ytr,a.random_state,a.n_estimators,a.n_jobs)
    acc,f1=evaluate(m,Xte,yte)
    rows.append(["Baseline RF",acc,f1,(time.perf_counter()-t)/60])

    # MI
    t=time.perf_counter()
    mi=mutual_info_classif(Xtr,ytr,random_state=a.random_state)
    idx=np.argsort(mi)[::-1][:min(a.mi_features,Xtr.shape[1])]
    m=rf(Xtr[:,idx],ytr,a.random_state,a.n_estimators,a.n_jobs)
    acc,f1=evaluate(m,Xte[:,idx],yte)
    rows.append(["MI + RF",acc,f1,(time.perf_counter()-t)/60])

    # PCA alone
    t=time.perf_counter()
    sc=StandardScaler().fit(Xtr)
    tr=sc.transform(Xtr); te=sc.transform(Xte)
    pca=PCA(n_components=min(a.pca_components,tr.shape[1]),random_state=a.random_state).fit(tr)
    ztr=pca.transform(tr); zte=pca.transform(te)
    m=rf(ztr,ytr,a.random_state,a.n_estimators,a.n_jobs)
    acc,f1=evaluate(m,zte,yte)
    rows.append(["PCA + RF",acc,f1,(time.perf_counter()-t)/60)

    # Sampling alone: cluster original feature space.
    t=time.perf_counter()
    reps=reps_from_kmeans(Xtr,a.clusters,a.random_state)
    m=rf(Xtr[reps],ytr[reps],a.random_state,a.n_estimators,a.n_jobs)
    acc,f1=evaluate(m,Xte,yte)
    rows.append(["Sampling + RF",acc,f1,(time.perf_counter()-t)/60)

    # MI + PCA
    t=time.perf_counter()
    ztr=StandardScaler().fit_transform(Xtr[:,idx])
    scaler=StandardScaler().fit(Xtr[:,idx])
    ztr=scaler.transform(Xtr[:,idx]); zte=scaler.transform(Xte[:,idx])
    pca=PCA(n_components=min(a.pca_components,ztr.shape[1]),random_state=a.random_state).fit(ztr)
    ztr=pca.transform(ztr); zte=pca.transform(zte)
    m=rf(ztr,ytr,a.random_state,a.n_estimators,a.n_jobs)
    acc,f1=evaluate(m,zte,yte)
    rows.append(["MI + PCA + RF",acc,f1,(time.perf_counter()-t)/60)

    # Complete CDLF
    t=time.perf_counter()
    reps=reps_from_kmeans(ztr,a.clusters,a.random_state)
    m=rf(ztr[reps],ytr[reps],a.random_state,a.n_estimators,a.n_jobs)
    acc,f1=evaluate(m,zte,yte)
    rows.append(["CDLF",acc,f1,(time.perf_counter()-t)/60])

    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows,columns=["Configuration","Accuracy (%)","F1-score (%)","Training Time (min)"])
    df.to_csv(out,index=False)
    print(df.to_string(index=False))
    print("Saved:",out.resolve())

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--mi-features",type=int,default=180)
    p.add_argument("--pca-components",type=int,default=50)
    p.add_argument("--clusters",type=int,default=50000)
    p.add_argument("--test-size",type=float,default=.20)
    p.add_argument("--random-state",type=int,default=42)
    p.add_argument("--n-estimators",type=int,default=300)
    p.add_argument("--n-jobs",type=int,default=-1)
    p.add_argument("--out",default="outputs/actual/table5_ablation.csv")
    main(p.parse_args())
