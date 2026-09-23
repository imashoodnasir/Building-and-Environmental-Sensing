
from common import *;from train_utils import *;import torch,numpy as np,json
def run():
 c=cfg();dev=device(c);A=torch.load(ROOT/"data/processed/target_graph.pt",weights_only=True).to(dev);m=build(c).to(dev);p=ROOT/"outputs/checkpoints/continual.pt";m.load_state_dict(torch.load(p if p.exists() else ROOT/"outputs/checkpoints/source.pt",map_location=dev,weights_only=True));m.eval();res={}
 for name,drop,noise in [("clean",0,0),("missing_modality",.25,0),("sensor_noise",0,.1),("combined",.25,.1)]:
  es=[]
  with torch.no_grad():
   for x,ma,y,ym in loader("target_deployment",c):
    x,ma,y,ym=[z.to(dev) for z in [x,ma,y,ym]]
    if drop: ma=ma*(torch.rand_like(ma)>drop);x=x*ma
    if noise:x=x+noise*torch.randn_like(x)*ma
    mu,sg,_,_=m(x,ma,A,"target");_,e=lossfn(mu,sg,y,ym,torch.tensor([1,1,1,0,1.],device=dev));es.append(e.item())
  res[name]=float(np.mean(es))
 json.dump(res,open(ROOT/"outputs/results/robustness.json","w"),indent=2);print(res)
if __name__=="__main__":run()
