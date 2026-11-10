"""Per-instance VLM pass over a label set: is each labelled object a cell?

The pairwise judge (vlm_judge.py) only sees where two models disagree, so a
false cell both models draw is never questioned.  This pass asks about every
instance in a label set on its own and turns the answers into an edited
label with a don't-care channel:

  cell          keep the instance
  not_cell      delete it (the pixels become background, a full label)
  bad_outline   a cell is there but the outline is wrong -> don't-care
  unsure        -> don't-care
  confidence below --conf, whatever the verdict -> don't-care

Don't-care pixels are written as -1 in the label TIFF; data.py turns them
into a 4th label channel that zeroes the detection loss there.

Stages (cached under --out, re-runnable one at a time):
  mine    one row per instance -> instances.csv (crop box, area)
  render  RAW | CANDIDATE panels, candidate outlined in yellow, other
          instances of the same label thin grey -> crops/<iid>.png, plus
          the full image with the crop boxed -> crops/<iid>_ctx.png
  query   Anthropic API, JSON verdict per instance -> responses.jsonl
  report  verdict counts; with --human <dir>, agreement with human labels
  apply   write edited labels (+ image symlinks) to --apply_out

Usage:
  python scripts/instance_judge.py all --labels ~/datasets/judge_ft/x/train \\
      --out analyses/vlm_judge/x-inst --hint "..." \\
      --human ~/datasets/judge_ft/human/train --apply_out ~/datasets/judge_ft/y/train
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ucell.utils import clean_up_mask, pad_channel  # noqa: E402
from vlm_judge import (  # noqa: E402
    PRICES, _png_b64, compose, render_context, square_crop, stretch_rgb,
    upsample_nearest,
)

VERDICTS = ["cell", "not_cell", "bad_outline", "unsure"]

SYSTEM_PROMPT = """\
You are an expert microscopist auditing a cell instance segmentation one
object at a time.

Each image has two panels, left to right:
  RAW        - a small crop of a microscopy image, contrast-stretched and
               magnified.
  CANDIDATE  - the same crop with ONE candidate object outlined in yellow.
               Other objects of the same segmentation, if any, are drawn as
               thin grey outlines; they are not being judged.

Decide about the yellow outline only:
  cell         - it encloses exactly one real cell and follows its boundary
                 reasonably well.
  not_cell     - there is no cell under it: background, substrate texture,
                 debris, culture-vessel wall, scratch, or out-of-focus junk.
  bad_outline  - a cell is there, but the outline is wrong: it covers two or
                 more cells, only a fragment of a cell, or is badly misplaced.
  unsure       - you cannot tell.
Ignore small boundary jitter.  Judge the object, not the rest of the crop.
"""

CONTEXT_PROMPT = (
    "The first image is the entire field of view with the judged crop boxed "
    "in yellow; use it to see what the cells in this sample look like and "
    "what is background, debris or culture-vessel structure.  The second "
    "image is the two-panel crop to judge."
)

USER_PROMPT = "Is the yellow outline a correctly segmented cell?  Reply with JSON only."

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "confidence": {"type": "number", "description": "0 to 1"},
        "reason": {"type": "string", "description": "under 30 words"},
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}

YELLOW = (255, 230, 0)
GREY = (150, 150, 150)


def label_files(d):
    return sorted(Path(d).glob("*_label.tif"))


def image_of(lf):
    return lf.with_name(lf.name.replace("_label", ""))


# --------------------------------------------------------------------- mine
def mine(args):
    rows = []
    for lf in label_files(args.labels):
        lab = tifffile.imread(lf).astype(np.int64)
        for p in regionprops(np.maximum(lab, 0)):
            y0, x0, y1, x1 = p.bbox
            cy0, cy1, cx0, cx1 = square_crop((y0, y1, x0, x1), lab.shape)
            rows.append(dict(
                image=image_of(lf).name, inst=int(p.label), area=int(p.area),
                y0=cy0, y1=cy1, x0=cx0, x1=cx1,
                crop_px=max(cy1 - cy0, cx1 - cx0),
                n_other=int((np.unique(lab[cy0:cy1, cx0:cx1]) > 0).sum() - 1),
            ))
    df = pd.DataFrame(rows)
    df.insert(0, "iid", [f"{r.image.replace('.tif', '')}-{r.inst:03d}"
                         for r in df.itertuples()])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "instances.csv", index=False)
    print(f"{len(df)} instances in {df.image.nunique()} images -> "
          f"{out / 'instances.csv'}")


# ------------------------------------------------------------------- render
def outline(rgb, mask, colour, thick):
    out = rgb.copy()
    bnd = find_boundaries(mask, mode="inner") & mask
    if thick > 1:
        bnd = ndi.binary_dilation(bnd, iterations=thick - 1) & mask
    out[bnd] = colour
    return out


def render(args):
    df = pd.read_csv(Path(args.out) / "instances.csv")
    out = Path(args.out) / "crops"
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    for row in df.itertuples():
        if row.image not in cache:
            lf = Path(args.labels) / row.image.replace(".tif", "_label.tif")
            img = tifffile.imread(image_of(lf))
            cache = {row.image: (pad_channel(img.astype(np.float32)),
                                 tifffile.imread(lf).astype(np.int32))}
        rgb, lab = cache[row.image]
        sl = (slice(row.y0, row.y1), slice(row.x0, row.x1))
        crop = stretch_rgb(rgb[sl])
        h, w = crop.shape[:2]
        s = args.panel_px / max(h, w)
        hw = (int(round(h * s)), int(round(w * s)))
        big = np.asarray(Image.fromarray(crop).resize(hw[::-1], Image.BICUBIC))
        gl = upsample_nearest(lab[sl], hw)
        pan = outline(big, (gl > 0) & (gl != row.inst), GREY, 1)
        pan = outline(pan, gl == row.inst, YELLOW, args.line_px)
        compose([big, pan], ["RAW", "CANDIDATE"]).save(out / f"{row.iid}.png")
        if args.context_px:
            render_context(rgb, row, args, out / f"{row.iid}_ctx.png")
    print(f"rendered {len(df)} crops under {out}")


# -------------------------------------------------------------------- query
def _img(path):
    return {"type": "image", "source": {"type": "base64",
                                        "media_type": "image/png",
                                        "data": _png_b64(path)}}


def parse(text):
    try:
        ans = json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", text, re.S)
        ans = json.loads(m.group(0)) if m else {}
    if not isinstance(ans, dict):
        ans = {}
    v = str(ans.get("verdict", "")).strip().lower()
    v = v if v in VERDICTS else "unsure"
    try:
        conf = float(ans.get("confidence", np.nan))
    except (TypeError, ValueError):
        conf = np.nan
    return dict(verdict=v, confidence=conf,
                reason=str(ans.get("reason", ""))[:300], raw=text[:300])


def ask(client, png, args):
    content = []
    ctx = png.with_name(png.stem + "_ctx.png")
    if args.context and ctx.exists():
        content += [{"type": "text", "text": CONTEXT_PROMPT}, _img(ctx)]
    hint = f"About this sample: {args.hint.strip()}\n\n" if args.hint else ""
    content += [_img(png), {"type": "text", "text": hint + USER_PROMPT}]
    resp = client.messages.create(
        model=args.model, max_tokens=args.max_tokens,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        output_config={"effort": args.effort,
                       "format": {"type": "json_schema",
                                  "schema": ANSWER_SCHEMA}},
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    u = resp.usage
    return dict(
        parse(text), stop_reason=resp.stop_reason,
        request_id=resp._request_id,
        input_tokens=u.input_tokens, output_tokens=u.output_tokens,
        cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def query(args):
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
    client = anthropic.Anthropic(max_retries=5)
    out = Path(args.out)
    df = pd.read_csv(out / "instances.csv")
    log = out / "responses.jsonl"
    done = set()
    if log.exists():
        for line in log.open():
            r = json.loads(line)
            done.add((r["iid"], r["model"], r["effort"]))
    jobs = [r.iid for r in df.itertuples()
            if (r.iid, args.model, args.effort) not in done]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(done)} cached, {len(jobs)} to query with {args.model} "
          f"effort={args.effort} context={args.context} workers={args.workers}")
    if args.dry_run or not jobs:
        return
    tot_in = tot_out = 0
    t0 = time.time()
    with log.open("a") as fh, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(ask, client, out / "crops" / f"{iid}.png", args): iid
                for iid in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            iid = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  {iid}: {type(e).__name__}: {e}")
                continue
            r.update(iid=iid, model=args.model, effort=args.effort,
                     ts=time.time())
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            tot_in += r["input_tokens"] + r["cache_read"] + r["cache_write"]
            tot_out += r["output_tokens"]
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s  "
                      f"in={tot_in} out={tot_out}")
    pin, pout = PRICES.get(args.model, (0, 0))
    print(f"tokens in={tot_in} out={tot_out}  approx cost "
          f"${tot_in * pin / 1e6 + tot_out * pout / 1e6:.2f}")


# ------------------------------------------------------------------- report
def load_verdicts(args):
    out = Path(args.out)
    df = pd.read_csv(out / "instances.csv").set_index("iid")
    rs = [json.loads(line) for line in (out / "responses.jsonl").open()]
    r = pd.DataFrame(rs)
    r = r[r.model == args.model]
    if args.effort:
        r = r[r.effort == args.effort]
    r = r.drop_duplicates("iid", keep="last").set_index("iid")
    d = df.join(r[["verdict", "confidence", "reason"]], how="left")
    d["verdict"] = d.verdict.fillna("unanswered")
    d["action"] = np.where(
        d.verdict.eq("cell") & (d.confidence >= args.conf), "keep",
        np.where(d.verdict.eq("not_cell") & (d.confidence >= args.conf),
                 "delete", "ignore"))
    return d


def human_match(d, args):
    """What the human label says about each instance: 'cell' if a human
    instance overlaps it with IoU >= 0.5, 'partial' if any human instance
    overlaps at all (merge / fragment), else 'not_cell'."""
    says = {}
    for image, g in d.groupby("image"):
        lab = tifffile.imread(Path(args.labels) / image.replace(".tif", "_label.tif")).astype(np.int64)
        hum = tifffile.imread(Path(args.human) / image.replace(".tif", "_label.tif")).astype(np.int64)
        for iid, r in g.iterrows():
            m = lab == r.inst
            ids, cnt = np.unique(hum[m], return_counts=True)
            keep = ids > 0
            if not keep.any():
                says[iid] = "not_cell"
                continue
            hid = ids[keep][cnt[keep].argmax()]
            inter = cnt[keep].max()
            union = m.sum() + (hum == hid).sum() - inter
            says[iid] = "cell" if inter / union >= 0.5 else "partial"
    return pd.Series(says)


def report(args):
    d = load_verdicts(args)
    out = Path(args.out)
    print("verdicts:", d.verdict.value_counts().to_dict())
    print("actions: ", d.action.value_counts().to_dict())
    print("median confidence by verdict:",
          d.groupby("verdict").confidence.median().round(2).to_dict())
    if args.human:
        d["human"] = human_match(d, args)
        print("\nhuman label vs verdict (rows human, cols verdict):")
        print(pd.crosstab(d.human, d.verdict))
        print("\nhuman label vs action:")
        print(pd.crosstab(d.human, d.action))
        dec = d[d.action != "ignore"]
        ok = ((dec.action == "keep") & (dec.human == "cell")) | \
             ((dec.action == "delete") & (dec.human == "not_cell"))
        print(f"\ndecided instances: {len(dec)}/{len(d)}, agreement with "
              f"human on decided: {ok.mean():.3f}  (human 'partial' counted "
              f"as disagreement)")
    d.to_csv(out / "verdicts.csv")
    per = d.groupby("image").action.value_counts().unstack(fill_value=0)
    per.to_csv(out / "per_image.csv")
    print(f"\nwrote {out / 'verdicts.csv'}")


# -------------------------------------------------------------------- apply
def apply(args):
    d = load_verdicts(args)
    out = Path(args.apply_out)
    out.mkdir(parents=True, exist_ok=True)
    n = dict(keep=0, delete=0, ignore=0, prior_ignore_px=0)
    for lf in label_files(args.labels):
        lab = tifffile.imread(lf).astype(np.int64)
        n["prior_ignore_px"] += int((lab < 0).sum())
        new = lab.copy()
        g = d[d.image == image_of(lf).name]
        for _, r in g.iterrows():
            m = lab == r.inst
            if r.action == "delete":
                new[m] = 0
            elif r.action == "ignore":
                # the object and a small rim: the boundary is uncertain too
                rim = ndi.binary_dilation(m, iterations=args.rim_px)
                new[m | (rim & (new == 0))] = -1
            n[r.action] += 1
        pos = clean_up_mask(np.maximum(new, 0))
        pos[new < 0] = -1
        tifffile.imwrite(out / lf.name, pos.astype(np.int32))
        link = out / image_of(lf).name
        if not link.exists():
            os.symlink(image_of(lf).resolve(), link)
    print("applied:", n, f"-> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["mine", "render", "query", "report",
                                     "apply", "all"])
    p.add_argument("--labels", required=True,
                   help="dir of <stem>.tif + <stem>_label.tif to audit")
    p.add_argument("--out", required=True)
    p.add_argument("--human", default=None,
                   help="report: dir of human labels for the same images")
    p.add_argument("--apply_out", default=None)
    p.add_argument("--panel_px", type=int, default=448)
    p.add_argument("--line_px", type=int, default=2)
    p.add_argument("--context_px", type=int, default=1024)
    p.add_argument("--context", action="store_true", default=True)
    p.add_argument("--no_context", dest="context", action="store_false")
    p.add_argument("--hint", default=None)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--conf", type=float, default=0.7)
    p.add_argument("--rim_px", type=int, default=3)
    args = p.parse_args()
    stages = (["mine", "render", "query", "report", "apply"]
              if args.stage == "all" else [args.stage])
    if "apply" in stages and not args.apply_out:
        p.error("apply needs --apply_out")
    for s in stages:
        globals()[s](args)


if __name__ == "__main__":
    main()
