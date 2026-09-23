
from common import *;from train_utils import *;import torch,numpy as np,pandas as pd,json,time
from sklearn.metrics import r2_score
def run():
 c=cfg();dev=device(c);A=torch.load(ROOT/"data/processed/target_graph.pt",weights_only=True).to(dev);m=build(c).to(dev);p=ROOT/"outputs/checkpoints/continual.pt";m.load_state_dict(torch.load(p if p.exists() else ROOT/"outputs/checkpoints/source.pt",map_location=dev,weights_only=True));m.eval()
 rows=[];lat=[]
 with torch.no_grad():
  for x,ma,y,ym in loader("target_deployment",c):
   x,ma,y,ym=[z.to(dev) for z in [x,ma,y,ym]];t=time.perf_counter();mu,sg,_,_=m(x,ma,A,"target");lat.append((time.perf_counter()-t)*1000/len(x))
   for f,n in enumerate(["temperature","relative_humidity","pressure","energy","pm25"]):
    mask=ym[...,f].bool().cpu().numpy();a=y[...,f].cpu().numpy()[mask];b=mu[...,f].cpu().numpy()[mask];ss=sg[...,f].cpu().numpy()[mask]
    if len(a):rows.append([n,np.abs(a-b).mean(),np.sqrt(((a-b)**2).mean()),100*np.mean(2*np.abs(a-b)/(np.abs(a)+np.abs(b)+1e-8)),np.mean((a>=b-1.96*ss)&(a<=b+1.96*ss)),np.mean(3.92*ss)])
 df=pd.DataFrame(rows,columns=["target","mae","rmse","smape","picp95","mpiw95"]).groupby("target").mean();df.to_csv(ROOT/"outputs/results/final_metrics.csv")
 pars=sum(p.numel() for p in m.parameters());summary={"parameters":pars,"latency_ms_per_window":float(np.mean(lat)),"throughput_windows_s":float(1000/np.mean(lat))}
 json.dump(summary,open(ROOT/"outputs/results/efficiency.json","w"),indent=2);print(df);print(summary)
if __name__=="__main__":run()
