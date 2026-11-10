"""Compose pseudo-labels from VLM judge verdicts.

Input is a vlm_judge run made with --ref_run: the reference model's mask is
the base label, and every judged disagreement where the judge consistently
and confidently preferred the other model (`pred`, i.e. --run) is applied as
an edit to that base:

  spurious  pred has an instance the reference lacks  -> add pred's instance
  missed    reference has an instance pred lacks      -> delete it
  merge     one pred instance covers >= 2 reference   -> replace them with it
  split     one reference covers >= 2 pred instances  -> replace it with them
  shape     one-to-one, poor overlap                  -> replace with pred's

Unjudged disagreements keep the reference's version.  Judged pairs with no
confident, order-consistent verdict either way also keep it, unless
--ignore_undecided is given: then the pixels of both proposals become
don't-care (-1 in the label TIFF; data.py zeroes the loss there), since
neither candidate is known to be right.  Pixels are written in event
order, so a later edit can overwrite an earlier one.

Usage:
  python scripts/judge_compose.py --judge_out analyses/vlm_judge/<run> \
      --run 375-pious-sound --ref_run cpsam_v2 --subset musc \
      --images ~/datasets/judge_ft/train_images.txt \
      --out ~/datasets/judge_ft/judge/train [--conf 0.7]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucell.utils import clean_up_mask  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge_out", required=True)
    p.add_argument("--run", required=True, help="the 'pred' model")
    p.add_argument("--ref_run", required=True, help="the base model")
    p.add_argument("--subset", required=True)
    p.add_argument("--tid", type=int, default=0)
    p.add_argument("--data", default=os.path.expanduser("~/datasets/scs_val"))
    p.add_argument("--images", required=True, help="text file of image names")
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.7)
    p.add_argument("--ignore_undecided", action="store_true",
                   help="mark judged-but-undecided pairs don't-care (-1)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    j = pd.read_csv(Path(args.judge_out) / "judgements.csv", index_col=0)
    if "gt_ids" not in j:
        pairs = pd.read_csv(Path(args.judge_out) / "pairs.csv").set_index("pair_id")
        j = j.join(pairs[["gt_ids", "pred_ids"]])
    sure = j.consistent & j.decided & (j.conf >= args.conf)
    apply = j[sure & (j.pick0 == "pred")]
    undecided = j[~sure]
    print(f"{len(j)} judged, {len(apply)} edits toward {args.run} "
          f"({apply.event.value_counts().to_dict()}), "
          f"{len(undecided)} undecided"
          + (" -> don't-care" if args.ignore_undecided else " -> keep ref"))

    names = Path(args.images).read_text().split()
    img_dir = Path(args.data) / args.subset
    n_edit = {e: 0 for e in ["spurious", "missed", "merge", "split", "shape"]}
    n_ignore = n_ignore_px = 0
    for n in names:
        stem = n.replace(".tif", "")
        ref = tifffile.imread(Path("predictions") / f"{args.ref_run}_{args.subset}"
                              / f"{stem}_mask.tif").astype(np.int64)
        pred = tifffile.imread(Path("predictions") / f"{args.run}_{args.subset}_t{args.tid}"
                               / f"{stem}_mask.tif").astype(np.int64)
        lab = ref.copy()
        next_id = int(lab.max()) + 1
        for _, r in apply[apply.image == n].iterrows():
            gids = [int(x) for x in str(r.gt_ids).split()] if pd.notna(r.gt_ids) else []
            pids = [int(x) for x in str(r.pred_ids).split()] if pd.notna(r.pred_ids) else []
            if gids:
                lab[np.isin(ref, gids) & np.isin(lab, gids)] = 0   # drop ref's
            for pid in pids:                                        # add pred's
                lab[pred == pid] = next_id
                next_id += 1
            n_edit[r.event] += 1
        if args.ignore_undecided:
            for _, r in undecided[undecided.image == n].iterrows():
                gids = [int(x) for x in str(r.gt_ids).split()] if pd.notna(r.gt_ids) else []
                pids = [int(x) for x in str(r.pred_ids).split()] if pd.notna(r.pred_ids) else []
                m = np.isin(ref, gids) | np.isin(pred, pids)
                # only pixels no surviving instance claims; a kept neighbour
                # stays supervised
                lab[m & (lab == 0)] = -1
                lab[m & np.isin(lab, gids)] = -1
                n_ignore += 1
        pos = clean_up_mask(np.maximum(lab, 0))
        pos[lab < 0] = -1
        n_ignore_px += int((lab < 0).sum())
        tifffile.imwrite(out / f"{stem}_label.tif",
                         pos.astype(np.int32 if (pos < 0).any() else np.uint16))
        link = out / n
        if not link.exists():
            os.symlink(img_dir / n, link)
    print("edits applied:", n_edit,
          f"; undecided pairs marked don't-care: {n_ignore} ({n_ignore_px} px)")
    print(f"wrote {len(names)} labels to {out}")


if __name__ == "__main__":
    main()
