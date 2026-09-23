
from common import *;from train_utils import *;import torch,numpy as np,json
def mmd(x,y,bw):
 def k(a,b):return torch.exp(-torch.cdist(a,b)**2/(2*bw*bw))
 return (k(x,x).mean()+k(y,y).mean()-2*k(x,y).mean()).clamp_min(0)
def reps(name,m,A,c,dev):
 out=[]
 with torch.no_grad():
  for x,ma,_,_ in loader(name,c):
   x,ma=x.to(dev),ma.to(dev);_,_,s,_=m(x,ma,A,"source" if name.startswith("source") else "target");out.append(s.mean(1).cpu())
 return torch.cat(out)
def run():
 c=cfg();dev=device(c);m=build(c).to(dev);m.load_state_dict(torch.load(ROOT/"outputs/checkpoints/source.pt",map_location=dev,weights_only=True));m.eval()
 As=torch.load(ROOT/"data/processed/source_graph.pt",weights_only=True).to(dev);At=torch.load(ROOT/"data/processed/target_graph.pt",weights_only=True).to(dev)
 r=reps("source_val",m,As,c,dev);d=torch.pdist(r);bw=float(torch.median(d[d>0])) if (d>0).any() else 1.
 n=min(c["mmd_reference_size"],len(r));ref=r[:n];scores=[]
 w=min(c["mmd_target_window"],max(4,len(r)//4))
 for i in range(w,len(r)-w,w):scores.append(float(mmd(ref,r[i:i+w],bw)))
 tau=float(np.quantile(scores,1-c["mmd_alpha"])) if scores else 0.
 json.dump({"bandwidth":bw,"threshold":tau},open(ROOT/"outputs/results/mmd.json","w"),indent=2);print("stage08",bw,tau)
if __name__=="__main__":run()
