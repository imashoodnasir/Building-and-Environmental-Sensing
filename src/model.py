
import torch,torch.nn as nn,torch.nn.functional as F
class Model(nn.Module):
    def __init__(self,c,nf=5):
        super().__init__();d=c["hidden_dim"];self.c=c
        self.enc=nn.ModuleList([nn.Sequential(nn.Linear(2,d),nn.ReLU(),nn.Linear(d,d)) for _ in range(nf)])
        self.score=nn.ModuleList([nn.Linear(d,1) for _ in range(nf)])
        self.g1=nn.Linear(d,d);self.g2=nn.Linear(d,d)
        self.att=nn.MultiheadAttention(d,c["temporal_heads"],batch_first=True,dropout=c["dropout"])
        self.shared=nn.Sequential(nn.Linear(d,d),nn.ReLU())
        self.ps=nn.Sequential(nn.Linear(d,d),nn.ReLU());self.pt=nn.Sequential(nn.Linear(d,d),nn.ReLU())
        self.gate=nn.Linear(2*d,d)
        self.mu=nn.ModuleList([nn.Linear(d,len(c["horizons"])) for _ in range(nf)])
        self.sg=nn.ModuleList([nn.Linear(d,len(c["horizons"])) for _ in range(nf)])
    def encode(self,x,m,A):
        # x,m B,L,N,F
        es=[];scores=[]
        for f in range(x.shape[-1]):
            e=self.enc[f](torch.stack([x[...,f],m[...,f]],-1));es.append(e);scores.append(self.score[f](e).squeeze(-1).masked_fill(m[...,f]<=0,-1e9))
        E=torch.stack(es,-2);S=torch.stack(scores,-1);w=torch.softmax(S,-1).unsqueeze(-1);u=(E*w).sum(-2)
        # static graph + latent dynamic similarity
        B,L,N,D=u.shape; last=F.normalize(u[:,-1],dim=-1);sim=torch.relu(last@last.transpose(1,2));AA=A[None]+sim/(sim.sum(-1,keepdim=True)+1e-6)
        z=[]
        for tt in range(L):
            h=torch.relu(self.g1(torch.einsum("bij,bjd->bid",AA,u[:,tt])))
            z.append(torch.relu(self.g2(torch.einsum("bij,bjd->bid",AA,h))))
        z=torch.stack(z,1) # B,L,N,D
        q=z.permute(0,2,1,3).reshape(B*N,L,D);q,_=self.att(q,q,q);h=q[:,-1].reshape(B,N,D)
        return h
    def forward(self,x,m,A,domain="source",stop_shared_pm=True):
        h=self.encode(x,m,A);s=self.shared(h);p=self.ps(h) if domain=="source" else self.pt(h)
        gate=torch.sigmoid(self.gate(torch.cat([s,p],-1)));q=gate*s+(1-gate)*p
        mus=[];sigs=[]
        for f in range(5):
            inp=q
            if domain=="target" and f==4 and stop_shared_pm:
                sd=s.detach();gate2=torch.sigmoid(self.gate(torch.cat([sd,p],-1)));inp=gate2*sd+(1-gate2)*p
            mus.append(self.mu[f](inp));sigs.append(F.softplus(self.sg[f](inp))+1e-4)
        return torch.stack(mus,-1),torch.stack(sigs,-1),s,p
