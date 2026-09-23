
from common import *;from train_utils import *;import torch,numpy as np,json
def run():
 c=cfg();dev=device(c);A=torch.load(ROOT/"data/processed/target_graph.pt",weights_only=True).to(dev);m=build(c).to(dev);m.load_state_dict(torch.load(ROOT/"outputs/checkpoints/source.pt",map_location=dev,weights_only=True));m.eval();vals=[]
 with torch.no_grad():
  for x,ma,y,ym in loader("target_calibration",c):
   x,ma=[z.to(dev) for z in [x,ma]];_,sg,_,_=m(x,ma,A,"target");vals.append(sg.cpu().numpy())
 a=np.concatenate(vals,0);thr=np.quantile(a,c["uncertainty_quantile"],axis=(0,1)).tolist();json.dump({"thresholds":thr},open(ROOT/"outputs/results/uncertainty_thresholds.json","w"));print("stage07 done")
if __name__=="__main__":run()
