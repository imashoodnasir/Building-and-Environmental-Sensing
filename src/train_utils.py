
from common import *
import pickle,torch,numpy as np
from torch.utils.data import Dataset,DataLoader
from model import Model
class DS(Dataset):
    def __init__(self,p):self.a=pickle.load(open(p,"rb"))
    def __len__(self):return len(self.a)
    def __getitem__(self,i):
        z=self.a[i];return torch.tensor(z["x"]),torch.tensor(z["mask"]),torch.tensor(z["y"]),torch.tensor(z["ymask"])
def lossfn(mu,sg,y,ym,eligible):
    mask=ym*eligible[None,None,None,:].to(ym.device);err=(y-mu)
    nll=(torch.log(sg)+.5*(err/sg)**2)*mask
    return nll.sum()/mask.sum().clamp_min(1), (err.abs()*mask).sum()/mask.sum().clamp_min(1)
def sep_loss(s,p): return ((s*s.mean((0,1),keepdim=True))*(p*p.mean((0,1),keepdim=True))).mean().abs()
def loader(name,c,shuffle=False):return DataLoader(DS(ROOT/f"data/processed/{name}_windows.pkl"),batch_size=c["batch_size"],shuffle=shuffle)
def build(c):return Model(c)
