"""Summarise the judge-signal fine-tune arms on the 40 held-out musc images.

Pooled F1 = 2*sum(tp) / (sum(n_pred) + sum(n_gt)); mean F1 is per-image.
Usage: python scripts/judge_ft_summary.py [analyses/judge_ft]
"""
import glob
import sys
from pathlib import Path

import pandas as pd

d = Path(sys.argv[1] if len(sys.argv) > 1 else "analyses/judge_ft")
rows = []
for f in sorted(glob.glob(str(d / "*_run001.csv"))):
    arm = Path(f).name.split("_5shot")[0].split("_run001")[0]
    r = pd.read_csv(f)
    tp, npred, ngt = r.n_tps.sum(), r.n_preds.sum(), r.n_gts.sum()
    rows.append(dict(arm=arm, n_pred=int(npred), n_gt=int(ngt), tp=int(tp),
                     pooled_f1=2 * tp / (npred + ngt), mean_f1=r.f1.mean(),
                     mean_ap=r.ap.mean(), pred_dice=r.pred_dice.mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
