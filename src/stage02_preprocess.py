
from common import *
import pandas as pd,numpy as np,json
def split(df,a,b):
    tt=sorted(df.timestamp.unique());i=int(len(tt)*a);j=int(len(tt)*(a+b))
    return df[df.timestamp<tt[i]].copy(),df[(df.timestamp>=tt[i])&(df.timestamp<tt[j])].copy(),df[df.timestamp>=tt[j]].copy()
def fit(df,vars):
    d={}
    for v in vars:
        x=df.loc[df["mask_"+v].eq(1),v].dropna().to_numpy(float)
        if v=="energy":x=np.log1p(np.maximum(x,0))
        d[v]={"mean":float(x.mean()),"std":float(x.std()+1e-8)}
    return d
def trans(df,st,target=False):
    o=df.copy()
    for v in ["temperature","relative_humidity","pressure"]+(["pm25"] if target else ["energy"]):
        m=o["mask_"+v].eq(1);x=o[v].astype(float)
        if v in ["energy","pm25"]:x=np.log1p(np.maximum(x,0))
        if v=="pm25":
            # causal expanding stats, shifted by one row within entity
            mu=o.assign(_x=x).groupby("entity_id")["_x"].transform(lambda z:z.expanding().mean().shift(1))
            sd=o.assign(_x=x).groupby("entity_id")["_x"].transform(lambda z:z.expanding().std(ddof=0).shift(1)).clip(lower=1e-6)
            z=(x-mu)/sd; z=z.fillna(0)
        else:z=(x-st[v]["mean"])/st[v]["std"]
        o[v+"_scaled"]=np.where(m,z,0.0)
    if target:o["energy_scaled"]=0.
    else:o["pm25_scaled"]=0.
    return o
def run():
    c=cfg();s=pd.read_csv(ROOT/"data/processed/source_hourly.csv",parse_dates=["timestamp"]);t=pd.read_csv(ROOT/"data/processed/target_hourly.csv",parse_dates=["timestamp"])
    tr,va,te=split(s,c["source_train_ratio"],c["source_val_ratio"]); st=fit(tr,c["shared"]+c["source_private"])
    tr,va,te=[trans(x,st) for x in [tr,va,te]]; t=trans(t,st,True)
    cut=sorted(t.timestamp.unique())[int(len(t.timestamp.unique())*c["target_calibration_ratio"])]
    tc=t[t.timestamp<cut];td=t[t.timestamp>=cut]
    for n,x in [("source_train",tr),("source_val",va),("source_test",te),("target_calibration",tc),("target_deployment",td)]:x.to_csv(ROOT/f"data/processed/{n}.csv",index=False)
    json.dump(st,open(ROOT/"data/processed/scalers.json","w"),indent=2);print("stage02 done")
if __name__=="__main__":run()
