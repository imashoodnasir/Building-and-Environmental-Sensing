
from common import *
import pandas as pd,numpy as np, pickle
VARS=["temperature","relative_humidity","pressure","energy","pm25"]
def make(path,domain):
    c=cfg();df=pd.read_csv(path,parse_dates=["timestamp"]); ents=sorted(df.entity_id.unique()); times=sorted(df.timestamp.unique())
    grid={}
    for e in ents:
        g=df[df.entity_id==e].set_index("timestamp").reindex(times)
        X=np.stack([g.get(v+"_scaled",pd.Series(0,index=g.index)).fillna(0).values for v in VARS],-1)
        M=np.stack([g.get("mask_"+v,pd.Series(0,index=g.index)).fillna(0).values for v in VARS],-1)
        Y=np.stack([g[v].values for v in VARS],-1)
        grid[e]=(X,M,Y)
    samples=[];L=c["lookback"];H=c["horizons"]
    for i in range(L-1,len(times)-max(H)):
        x=np.stack([grid[e][0][i-L+1:i+1] for e in ents],1) # L,N,F
        m=np.stack([grid[e][1][i-L+1:i+1] for e in ents],1)
        y=np.stack([[grid[e][2][i+h] for h in H] for e in ents],0) # N,K,F
        ym=np.isfinite(y).astype(np.float32);y=np.nan_to_num(y).astype(np.float32)
        samples.append({"x":x.astype(np.float32),"mask":m.astype(np.float32),"y":y,"ymask":ym,"origin":str(times[i]),"entities":ents,"domain":domain})
    return samples
def run():
    for n,d in [("source_train","source"),("source_val","source"),("source_test","source"),("target_calibration","target"),("target_deployment","target")]:
        s=make(ROOT/f"data/processed/{n}.csv",d);pickle.dump(s,open(ROOT/f"data/processed/{n}_windows.pkl","wb"));print(n,len(s))
if __name__=="__main__":run()
