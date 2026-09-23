
from common import *;from train_utils import *;import torch,numpy as np,json
def run():
 c=cfg();dev=device(c);A=torch.load(ROOT/"data/processed/target_graph.pt",weights_only=True).to(dev);m=build(c).to(dev);m.load_state_dict(torch.load(ROOT/"outputs/checkpoints/source.pt",map_location=dev,weights_only=True));m.eval()
 elig=torch.tensor([1,1,1,0,0.],device=dev);errs=[]
 with torch.no_grad():
  for x,ma,y,ym in loader("target_deployment",c):
   x,ma,y,ym=[z.to(dev) for z in [x,ma,y,ym]];mu,sg,_,_=m(x,ma,A,"target");_,e=lossfn(mu,sg,y,ym,elig);errs.append(e.item())
 r={"zero_shot_mae":float(np.mean(errs))};json.dump(r,open(ROOT/"outputs/results/zero_shot.json","w"),indent=2);print(r)
if __name__=="__main__":run()
