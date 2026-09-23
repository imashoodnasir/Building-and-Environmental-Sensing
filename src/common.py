
from pathlib import Path
import random, yaml, numpy as np, torch

ROOT=Path(__file__).resolve().parents[1]
def cfg():
    with open(ROOT/"config.yaml","r",encoding="utf8") as f:return yaml.safe_load(f)
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def device(c):
    return torch.device("cuda" if c.get("device")=="auto" and torch.cuda.is_available() else ("cpu" if c.get("device")=="auto" else c["device"]))
def ensure():
    for p in ["data/processed","outputs/checkpoints","outputs/results"]:(ROOT/p).mkdir(parents=True,exist_ok=True)
