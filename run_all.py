
import sys,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
stages=["stage01_prepare.py","stage02_preprocess.py","stage03_windows.py","stage04_graphs.py","stage05_pretrain.py","stage06_zero_shot.py","stage07_uncertainty.py","stage08_shift.py","stage09_continual.py","stage10_robustness.py","stage11_ablations.py","stage12_evaluate.py"]
def call(x): subprocess.run([sys.executable,str(ROOT/"src"/x)],check=True,cwd=ROOT)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--demo",action="store_true");p.add_argument("--full-sweep",action="store_true");a=p.parse_args()
 if a.demo: call("demo_data.py")
 for s in stages: call(s)
 print("\\nALL 12 STAGES COMPLETED.")
 if a.full_sweep: print("Full 20-seed/ablation sweep hook is intentionally not auto-run in this compact demo; use config seeds with your cluster launcher.")
