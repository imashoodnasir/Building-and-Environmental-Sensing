
from common import *
import pandas as pd, numpy as np, re, json
ALIASES={
"timestamp":["timestamp","datetime","date_time","date","time","created_at"],
"entity_id":["entity_id","building_id","building","meter_id","station_id","station","sensor_id","box_id","id"],
"temperature":["temperature","air_temperature","temp","temperature_c"],
"relative_humidity":["relative_humidity","humidity","rh","air_humidity"],
"pressure":["pressure","air_pressure","atmospheric_pressure","pressure_hpa"],
"energy":["energy","energy_kwh","electricity","consumption","meter_reading"],
"pm25":["pm25","pm2_5","pm_2_5","particulate_matter_2_5"],
"latitude":["latitude","lat"],"longitude":["longitude","lon","lng"]}
def norm(s): return re.sub("_+","_",re.sub("[^a-z0-9]+","_",str(s).lower())).strip("_")
def detect(cols,k):
    d={norm(x):x for x in cols}
    for a in ALIASES[k]:
        if norm(a) in d:return d[norm(a)]
def canon(df,domain,default):
    o=pd.DataFrame(); tc=detect(df.columns,"timestamp")
    if tc is None: raise ValueError("timestamp not detected")
    o["timestamp"]=pd.to_datetime(df[tc],errors="coerce",utc=True)
    ec=detect(df.columns,"entity_id"); o["entity_id"]=df[ec].astype(str) if ec else default
    for v in ["temperature","relative_humidity","pressure","energy","pm25","latitude","longitude"]:
        cc=detect(df.columns,v); o[v]=pd.to_numeric(df[cc],errors="coerce") if cc else np.nan
    if o.temperature.median(skipna=True)>150:o.temperature-=273.15
    if o.pressure.median(skipna=True)>2000:o.pressure/=100
    if 0<=o.relative_humidity.median(skipna=True)<=1.5:o.relative_humidity*=100
    o["domain"]=domain; return o.dropna(subset=["timestamp"])
def load(folder,domain):
    fs=list(folder.rglob("*.csv"))
    if not fs: raise FileNotFoundError(f"No CSV files in {folder}. Run --demo or add real files.")
    arr=[]
    for f in fs:
        try: arr.append(canon(pd.read_csv(f,low_memory=False),domain,f.stem))
        except Exception as e: print("skip",f,e)
    return pd.concat(arr,ignore_index=True)
def hourly(df):
    vals=["temperature","relative_humidity","pressure","energy","pm25"]
    out=[]
    for (d,e),g in df.groupby(["domain","entity_id"]):
        g=g.set_index("timestamp").sort_index()
        x=g[vals].resample("1h").mean()
        x["latitude"]=g.latitude.dropna().median(); x["longitude"]=g.longitude.dropna().median()
        x["domain"]=d;x["entity_id"]=e;out.append(x.reset_index())
    z=pd.concat(out,ignore_index=True)
    for v in vals:z["mask_"+v]=z[v].notna().astype("int8")
    return z
def run():
    ensure(); s=hourly(load(ROOT/"data/raw/bdg2","source")); t=hourly(load(ROOT/"data/raw/opensensemap","target"))
    s["pm25"]=np.nan;s["mask_pm25"]=0;t["energy"]=np.nan;t["mask_energy"]=0
    s.to_csv(ROOT/"data/processed/source_hourly.csv",index=False);t.to_csv(ROOT/"data/processed/target_hourly.csv",index=False)
    print("stage01",s.shape,t.shape)
if __name__=="__main__":run()
