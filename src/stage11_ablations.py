
from common import *;import json
# Runnable ablation launcher: records configurations; full retraining is optional because
# a publication-quality 20-seed sweep can be expensive.
ABLATIONS={"full":{},"no_modality_dropout":{"modality_dropout":0.0},"no_alignment":{"lambda_align":0.0},"no_replay":{"lambda_replay":0.0},"no_separation":{"lambda_sep":0.0}}
def run():
 out={"ablations":ABLATIONS,"note":"Use run_all.py --full-sweep to retrain each configuration over configured seeds."}
 json.dump(out,open(ROOT/"outputs/results/ablation_plan.json","w"),indent=2);print("stage11 done")
if __name__=="__main__":run()
