#!/bin/bash
#SBATCH --qos=general
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH --gres=gpu:L40S:1
#SBATCH -t 0:20:00
#SBATCH -J ignore_smoke
# Loads a label dir that holds -1 don't-care pixels through data.py on the
# GPU (flow computation), runs one FRMWrapper forward with the loss, and
# checks the 4th label channel and the weighted det_loss.
set -euo pipefail
cd "$HOME/work/ucell"
source .venv/bin/activate
python -u - <<'PY'
import numpy as np, torch, tifffile, glob
import data
from ucell.frm import FRMWrapper
from config import get_config
cfg = get_config("train")
cfg.data_dir = "/home/FCAM/jyu/datasets/judge_ft/_smoke"
cfg.task_id = 0; cfg.batch_size = 2; cfg.epochs_per_iter = 1
cfg.model.hidden_size = 1024; cfg.model.norm_task_emb = True
fs = sorted(glob.glob(cfg.data_dir + "/train/*_label.tif"))
neg = [f for f in fs if tifffile.imread(f).min() < 0]
print(len(fs), "labels,", len(neg), "with don't-care pixels")
ds = data.load_img_folder(cfg, "train")
ex = ds[0]
lab = np.stack([np.asarray(x) for x in ex["label"]])
print("label channels:", lab.shape, "ignore px:", int(lab[3].sum()), "seg px:", int(lab[2].sum()))
assert lab.shape[0] == 4
dl = data.get_dataloader(cfg, "train")
batch = next(iter(dl))
print("batch label:", tuple(batch["label"].shape), "ignore frac per sample:", batch["label"][:, 3].mean((1, 2)).tolist())
model = FRMWrapper(cfg).cuda()
model.load_checkpoint("checkpoints/375-pious-sound/final_ema.pt")
model.train()
batch = {k: (v.cuda().to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v.cuda()) for k, v in batch.items()}
carry = model.initial_carry(batch)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(carry, batch)
print("losses:", {k: v.float().tolist() for k, v in out["losses"].items()})
# the weighted det_loss must equal a plain mean when the ignore channel is cleared
batch2 = dict(batch); batch2["label"] = batch["label"].clone(); batch2["label"][:, 3] = 0
with torch.autocast("cuda", dtype=torch.bfloat16):
    out2 = model(model.initial_carry(batch2), batch2)
print("det_loss without ignore:", out2["losses"]["det_loss"].float().tolist())
print("OK")
PY
