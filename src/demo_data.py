
from common import *
import pandas as pd,numpy as np
def run(hours=360,nodes=4):
    seed_all(1); ensure()
    for domain in ["bdg2","opensensemap"]:
        rows=[]
        start=pd.Timestamp("2025-01-01",tz="UTC")
        for n in range(nodes):
            lat=54.8+n*.02;lon=23.8+n*.02
            for h in range(hours):
                ts=start+pd.Timedelta(hours=h); daily=np.sin(2*np.pi*h/24); noise=np.random.normal
                row={"timestamp":ts,"entity_id":f"{domain}_{n}","temperature":15+7*daily+noise(0,.5),
                     "relative_humidity":60-12*daily+noise(0,1.5),"pressure":1013+3*np.sin(2*np.pi*h/168)+noise(0,.5),
                     "latitude":lat,"longitude":lon}
                if domain=="bdg2": row["energy"]=max(0,25-5*daily+noise(0,1))
                else: row["pm25"]=max(0,18+5*np.sin(2*np.pi*h/72)+noise(0,2))
                rows.append(row)
        p=ROOT/f"data/raw/{domain}/demo.csv";pd.DataFrame(rows).to_csv(p,index=False);print("wrote",p)
if __name__=="__main__":run()
