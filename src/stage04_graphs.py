
from common import *
import pandas as pd,numpy as np,torch
def hav(lat1,lon1,lat2,lon2):
    p1,p2=np.radians(lat1),np.radians(lat2);dp=p2-p1;dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 6371*2*np.arcsin(np.sqrt(a))
def graph(path,k):
    df=pd.read_csv(path);meta=df.groupby("entity_id")[["latitude","longitude"]].median();n=len(meta);D=np.zeros((n,n))
    a=meta.to_numpy()
    for i in range(n):
        for j in range(n):
            if np.isfinite(a[[i,j]]).all():D[i,j]=hav(*a[i],*a[j])
            else:D[i,j]=abs(i-j)
    A=np.zeros_like(D)
    for i in range(n):
        ids=np.argsort(D[i]+np.eye(n)[i]*1e9)[:min(k,n-1)];A[i,ids]=np.exp(-D[i,ids]/(np.median(D[i,ids])+1e-6))
    A=(A+A.T)/2;A+=np.eye(n);deg=np.maximum(A.sum(1),1e-8);A=np.diag(deg**-.5)@A@np.diag(deg**-.5)
    return torch.tensor(A,dtype=torch.float32)
def run():
    c=cfg();torch.save(graph(ROOT/"data/processed/source_train.csv",c["knn"]),ROOT/"data/processed/source_graph.pt")
    torch.save(graph(ROOT/"data/processed/target_calibration.csv",c["knn"]),ROOT/"data/processed/target_graph.pt");print("stage04 done")
if __name__=="__main__":run()
