
from common import *;from train_utils import *;from stage08_shift import mmd
import torch,pickle,json,numpy as np
def run():
 c=cfg();dev=device(c);m=build(c).to(dev);m.load_state_dict(torch.load(ROOT/"outputs/checkpoints/source.pt",map_location=dev,weights_only=True));At=torch.load(ROOT/"data/processed/target_graph.pt",weights_only=True).to(dev);As=torch.load(ROOT/"data/processed/source_graph.pt",weights_only=True).to(dev)
 info=json.load(open(ROOT/"outputs/results/mmd.json"));bw=info["bandwidth"];tau=info["threshold"];opt=torch.optim.AdamW(m.parameters(),lr=c["learning_rate"]*.2)
 src=DS(ROOT/"data/processed/source_train_windows.pkl"); tgt=DS(ROOT/"data/processed/target_deployment_windows.pkl")
 # fixed historical reference
 ids=np.linspace(0,len(src)-1,min(c["mmd_reference_size"],len(src))).astype(int); rr=[]
 m.eval()
 with torch.no_grad():
  for i in ids:
   x,ma,_,_=src[i];_,_,s,_=m(x[None].to(dev),ma[None].to(dev),As,"source");rr.append(s.mean(1).squeeze(0))
 ref=torch.stack(rr);adapt=0;pending=[]
 for i in range(len(tgt)):
  x,ma,y,ym=tgt[i];x=x[None].to(dev);ma=ma[None].to(dev);y=y[None].to(dev);ym=ym[None].to(dev)
  m.eval()
  with torch.no_grad():_,_,s,_=m(x,ma,At,"target");score=float(mmd(ref,s.mean(1),bw))
  pending.append((x,ma,y,ym))
  if score>tau and len(pending)>=max(1,c["min_matured_labels"]//(len(c["horizons"])*3)):
   m.train();opt.zero_grad();L=0
   for xx,mm,yy,yym in pending[-4:]:
    mu,sg,ss,pp=m(xx,mm,At,"target");elig=torch.tensor([1,1,1,0,1.],device=dev);l,_=lossfn(mu,sg,yy,yym,elig);L=L+l+c["lambda_sep"]*sep_loss(ss,pp)
   L=L/len(pending[-4:]);L.backward();opt.step();pending=[];adapt+=1
 torch.save(m.state_dict(),ROOT/"outputs/checkpoints/continual.pt");json.dump({"adaptation_events":adapt},open(ROOT/"outputs/results/continual.json","w"));print("stage09",adapt)
if __name__=="__main__":run()
