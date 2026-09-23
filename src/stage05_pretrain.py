
from common import *;from train_utils import *;import torch
def run():
 c=cfg();seed_all(c["seed"]);dev=device(c);A=torch.load(ROOT/"data/processed/source_graph.pt",weights_only=True).to(dev);m=build(c).to(dev);o=torch.optim.AdamW(m.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"]);elig=torch.tensor([1,1,1,1,0.],device=dev)
 for ep in range(c["epochs"]):
  m.train();tot=0
  for x,ma,y,ym in loader("source_train",c,True):
   x,ma,y,ym=[z.to(dev) for z in [x,ma,y,ym]];mu,sg,s,p=m(x,ma,A,"source");L,_=lossfn(mu,sg,y,ym,elig);L=L+c["lambda_sep"]*sep_loss(s,p);o.zero_grad();L.backward();o.step();tot+=L.item()
  print("epoch",ep+1,tot)
 torch.save(m.state_dict(),ROOT/"outputs/checkpoints/source.pt");print("stage05 done")
if __name__=="__main__":run()
