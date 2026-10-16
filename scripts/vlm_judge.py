"""VLM-as-judge experiment for cell instance segmentation.

Question: on a localized crop where the ground-truth label and a ucell
prediction disagree at the instance level (merge / split / missed cell /
spurious cell / poor outline), can a vision-language model pick the correct
one?  If it can, coarse "which is better" judgements could come from an API
call instead of a human annotator.

The pipeline is four cached stages under --out (default
analyses/vlm_judge/<run>), so every stage can be re-run on its own:

  mine    pair up <data>/<subset>/*_label.tif with
          predictions/<run>_<subset>_t<tid>/*_mask.tif, find instance-level
          disagreement events, pick a square crop around each one and write
          pairs.csv (one row per event, with strata features).
  render  draw   RAW | A | B   panels (raw crop, candidate outlines) for both
          A/B orderings of every pair -> crops/<pair_id>_o{0,1}.png.
  query   send each PNG to the model, ask for a JSON verdict, append to
          responses.jsonl (already-answered items are skipped on re-run).
  report  score against ground truth: accuracy, order consistency, position
          bias, confidence gating, stratified by subset / event / contrast /
          density.  Writes report.csv and a sheet.html contact sheet.
  all     mine -> render -> query -> report.

The correct answer is by construction the ground-truth label, so the numbers
are a lower bound: where the GT itself is wrong the judge is penalised for
being right.

Usage:
  python scripts/vlm_judge.py mine   --run 375-pious-sound --n_per_subset 40
  python scripts/vlm_judge.py render --run 375-pious-sound
  ANTHROPIC_API_KEY=... python scripts/vlm_judge.py query --run 375-pious-sound
  python scripts/vlm_judge.py report --run 375-pious-sound

`query --limit 4` is a cheap smoke test of credentials and prompt.

Any OpenAI-compatible chat endpoint (OpenAI, vLLM, Ollama, OpenRouter, a
proxy) works through --backend openai:

  OPENAI_API_KEY=... OPENAI_BASE_URL=http://host:8000/v1 \\
      python scripts/vlm_judge.py query --backend openai --model <name>

Answers from different models share responses.jsonl; `report --model <name>`
scores one of them.
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucell.metrics import mask_intersection  # noqa: E402
from ucell.utils import pad_channel  # noqa: E402

SUBSETS = ["cellpose", "livecell", "nips", "tissuenet", "monusac_test"]
EVENTS = ["merge", "split", "missed", "spurious", "shape"]

# $/MTok (input, output), first-party API, for the cost line in `query`
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0),
}

PALETTE = [
    (255, 230, 0), (0, 230, 255), (255, 80, 255), (80, 255, 80),
    (255, 140, 0), (120, 160, 255), (255, 255, 255), (255, 60, 60),
]

SYSTEM_PROMPT = """\
You are an expert microscopist grading cell instance segmentations.

Each image has three panels, left to right:
  RAW - a small crop of a microscopy image (fluorescence, phase contrast,
        brightfield or histology), contrast-stretched and magnified.
  A   - the same crop with candidate segmentation A drawn as outlines.
  B   - the same crop with candidate segmentation B drawn as outlines.
Every closed coloured outline is one cell instance.  Touching cells are
drawn with different colours so you can tell them apart.  A cell with no
outline around it is not segmented by that candidate.

Decide which candidate is the more correct segmentation of the cells in the
crop.  Judge in this order of importance:
  1. Instance count: cells wrongly merged into one outline, one cell wrongly
     cut into several, visible cells with no outline, outlines where there is
     no cell.
  2. Outline placement: whether the outline follows the actual cell boundary.
Ignore small boundary jitter, and ignore everything outside the crop.  The
two candidates usually differ in only one place; find it first.  Answer
"tie" only if you cannot see a meaningful difference or cannot decide.
"""

USER_PROMPT = (
    "Which candidate segments the cells in this crop more correctly, A or "
    "B?  Reply with JSON only."
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        # no minimum/maximum: the Claude API rejects numeric bounds in
        # output_config.format, so the ranges live in the prompt
        "choice": {"type": "string", "enum": ["A", "B", "tie"]},
        "confidence": {"type": "number",
                       "description": "0 to 1, how sure you are"},
        "n_cells": {"type": "integer",
                    "description": "how many cells you see in the crop"},
        "reason": {"type": "string", "description": "under 30 words"},
    },
    "required": ["choice", "confidence", "n_cells", "reason"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------- mine
def find_events(gt, pred, min_area=30, dom=0.5, none=0.2, shape_iou=0.6):
    """Instance-level disagreement events between two label images.

    Returns a list of (event, gt_ids, pred_ids).  Fractions are of the
    *smaller* object's area inside the larger one's, so a merge is a pred
    that contains >= `dom` of each of two or more GT cells, and a split is a
    GT that contains >= `dom` of two or more preds.
    """
    gt_rps = regionprops(gt)
    pr_rps = regionprops(pred)
    if not gt_rps or not pr_rps:
        return []
    its = mask_intersection(pr_rps, gt_rps).astype(float)  # [n_pred, n_gt]
    pa = np.array([r.area_filled for r in pr_rps], float)
    ga = np.array([r.area_filled for r in gt_rps], float)
    frac_gt = its / ga[None, :]      # share of each GT inside pred p
    frac_pr = its / pa[:, None]      # share of each pred inside GT g
    iou = its / (pa[:, None] + ga[None, :] - its)

    events = []
    for p, rp in enumerate(pr_rps):
        gs = np.where(frac_gt[p] >= dom)[0]
        if len(gs) >= 2:
            events.append(("merge", [gt_rps[g].label for g in gs],
                           [rp.label]))
        elif frac_pr[p].max() < none and pa[p] >= min_area:
            events.append(("spurious", [], [rp.label]))
    for g, rg in enumerate(gt_rps):
        ps = np.where(frac_pr[:, g] >= dom)[0]
        if len(ps) >= 2:
            events.append(("split", [rg.label],
                           [pr_rps[p].label for p in ps]))
        elif frac_gt[:, g].max() < none and ga[g] >= min_area:
            events.append(("missed", [rg.label], []))
        else:
            p = int(iou[:, g].argmax())
            # one-to-one partner with a mediocre overlap: outline quality
            if (none <= iou[p, g] < shape_iou and ga[g] >= min_area
                    and (frac_gt[p] >= dom).sum() == 1
                    and (frac_pr[:, g] >= dom).sum() <= 1):
                events.append(("shape", [rg.label], [pr_rps[p].label]))
    return events


def union_bbox(gt, pred, gt_ids, pred_ids):
    m = np.isin(gt, gt_ids) if gt_ids else np.zeros_like(gt, bool)
    if pred_ids:
        m |= np.isin(pred, pred_ids)
    ys, xs = np.where(m)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def square_crop(bbox, shape, margin=0.6, min_px=48):
    y0, y1, x0, x1 = bbox
    h, w = y1 - y0, x1 - x0
    s = max(int(max(h, w) * (1 + 2 * margin)), min_px)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    H, W = shape
    s = min(s, max(H, W))
    ya = int(round(cy - s / 2))
    xa = int(round(cx - s / 2))
    ya = min(max(ya, 0), max(H - s, 0))
    xa = min(max(xa, 0), max(W - s, 0))
    return ya, min(ya + s, H), xa, min(xa + s, W)


def box_iou(a, b):
    ya = max(a[0], b[0]); yb = min(a[1], b[1])
    xa = max(a[2], b[2]); xb = min(a[3], b[3])
    inter = max(0, yb - ya) * max(0, xb - xa)
    area = lambda r: (r[1] - r[0]) * (r[3] - r[2])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


def to_gray(img):
    rgb = pad_channel(img.astype(np.float32))
    return rgb.mean(-1)


def balanced_sample(rows, k, rng):
    """Up to k rows, round-robin over event types."""
    by_ev = {e: [r for r in rows if r["event"] == e] for e in EVENTS}
    for e in by_ev:
        rng.shuffle(by_ev[e])
    picked = []
    while len(picked) < k and any(by_ev.values()):
        for e in EVENTS:
            if by_ev[e] and len(picked) < k:
                picked.append(by_ev[e].pop())
    return picked


def mine(args):
    rng = random.Random(args.seed)
    rows = []
    all_found = []
    for subset in args.subsets:
        data = Path(args.data) / subset
        pred_dir = Path("predictions") / f"{args.run}_{subset}_t{args.tid}"
        labels = sorted(data.glob("*_label.tif"))
        if not labels or not pred_dir.exists():
            print(f"[{subset}] skipped: {len(labels)} labels, "
                  f"pred dir exists={pred_dir.exists()}")
            continue
        rng.shuffle(labels)
        if not args.per_image:
            labels = labels[:args.max_images]
        found = []
        for lf in labels:
            gt = tifffile.imread(lf).astype(np.int64)
            if gt.min() < 0:   # has ignore regions; skip the image
                continue
            mf = pred_dir / lf.name.replace("_label", "_mask")
            if not mf.exists():
                continue
            pred = tifffile.imread(mf).astype(np.int64)
            img = tifffile.imread(lf.with_name(lf.name.replace("_label", "")))
            gray = to_gray(img)
            accepted = []
            for ev, gids, pids in find_events(gt, pred):
                if ev not in args.events:
                    continue
                bb = union_bbox(gt, pred, gids, pids)
                crop = square_crop(bb, gt.shape)
                if any(box_iou(crop, c) > 0.5 for c in accepted):
                    continue
                accepted.append(crop)
                y0, y1, x0, x1 = crop
                g = gray[y0:y1, x0:x1]
                gl = gt[y0:y1, x0:x1]
                n_gt = len(np.unique(gl)) - (1 if (gl == 0).any() else 0)
                n_pr = len(np.unique(pred[y0:y1, x0:x1]))
                n_pr -= 1 if (pred[y0:y1, x0:x1] == 0).any() else 0
                found.append(dict(
                    subset=subset, image=lf.name.replace("_label", ""),
                    event=ev, gt_ids=" ".join(map(str, gids)),
                    pred_ids=" ".join(map(str, pids)),
                    y0=y0, y1=y1, x0=x0, x1=x1, crop_px=max(y1 - y0, x1 - x0),
                    n_gt=n_gt, n_pred=n_pr,
                    density=1e4 * n_gt / max((y1 - y0) * (x1 - x0), 1),
                    contrast=float(g.std() / (g.mean() + 1e-6)),
                    gt_area=float((gl > 0).sum() / max(n_gt, 1)),
                ))
        if args.per_image:
            # every image gets up to K events, balanced across types, so
            # per-image verdict rates are comparable between images
            picked = []
            for img in sorted({r["image"] for r in found}):
                picked += balanced_sample(
                    [r for r in found if r["image"] == img],
                    args.per_image, rng)
        else:
            picked = balanced_sample(found, args.n_per_subset, rng)
        all_found += found
        counts = {e: sum(r["event"] == e for r in picked) for e in EVENTS}
        print(f"[{subset}] {len(labels)} images, {len(found)} events, "
              f"picked {len(picked)}: {counts}")
        rows += picked
    df = pd.DataFrame(rows)
    df.insert(0, "pair_id", [f"{r.subset}-{i:04d}" for i, r in
                             enumerate(df.itertuples())])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pairs.csv", index=False)
    # every mined event per image, before sampling: how *many*
    # disagreements an image has, which the sampled pairs no longer show
    pd.DataFrame(all_found).groupby(["subset", "image", "event"]).size() \
        .rename("n_events").reset_index() \
        .to_csv(out / "image_events.csv", index=False)
    print(f"wrote {out / 'pairs.csv'} ({len(df)} pairs)")


# ------------------------------------------------------------------- render
def stretch_rgb(img):
    """Crop -> uint8 RGB, per-channel percentile stretch, dark channels left
    dark so a 2-channel image keeps its red/green reading."""
    rgb = pad_channel(img.astype(np.float32)).copy()
    for c in range(3):
        ch = rgb[..., c]
        lo, hi = np.percentile(ch, [1, 99.5])
        if hi - lo < 1e-6:
            rgb[..., c] = 0 if hi < 1e-6 else 0.5
        else:
            rgb[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return (rgb * 255).astype(np.uint8)


def upsample_nearest(a, hw):
    h, w = a.shape[:2]
    yi = (np.arange(hw[0]) * h / hw[0]).astype(int)
    xi = (np.arange(hw[1]) * w / hw[1]).astype(int)
    return a[yi][:, xi]


def draw_outlines(rgb, label, thick=2):
    out = rgb.copy()
    bnd = find_boundaries(label, mode="inner") & (label > 0)
    if thick > 1:
        bnd = ndi.binary_dilation(bnd, iterations=thick - 1) & (label > 0)
    ids = label[bnd]
    cols = np.array(PALETTE, np.uint8)[(ids - 1) % len(PALETTE)]
    out[bnd] = cols
    return out


def font(size):
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def compose(panels, titles, gap=10, header=30):
    h, w = panels[0].shape[:2]
    W = len(panels) * w + (len(panels) - 1) * gap
    canvas = Image.new("RGB", (W, h + header), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    f = font(22)
    for i, (p, t) in enumerate(zip(panels, titles)):
        x = i * (w + gap)
        canvas.paste(Image.fromarray(p), (x, header))
        d.text((x + w // 2 - 6 * len(t), 4), t, fill=(255, 255, 255), font=f)
    return canvas


def render_pair(row, args):
    data = Path(args.data) / row.subset
    pred_dir = Path("predictions") / f"{args.run}_{row.subset}_t{args.tid}"
    img = tifffile.imread(data / f"{row.image}")
    gt = tifffile.imread(data / f"{row.image}".replace(".tif", "_label.tif"))
    pred = tifffile.imread(
        pred_dir / f"{row.image}".replace(".tif", "_mask.tif"))
    sl = (slice(row.y0, row.y1), slice(row.x0, row.x1))
    rgb = pad_channel(img.astype(np.float32))
    crop = stretch_rgb(rgb[sl])
    gl, pl = gt[sl].astype(np.int32), pred[sl].astype(np.int32)
    h, w = crop.shape[:2]
    s = args.panel_px / max(h, w)
    hw = (int(round(h * s)), int(round(w * s)))
    big = np.asarray(Image.fromarray(crop).resize(hw[::-1], Image.BICUBIC))
    gl, pl = upsample_nearest(gl, hw), upsample_nearest(pl, hw)
    pan_gt = draw_outlines(big, gl, args.line_px)
    pan_pr = draw_outlines(big, pl, args.line_px)
    out = Path(args.out) / "crops"
    out.mkdir(parents=True, exist_ok=True)
    # ordering 0: A = ground truth; ordering 1: A = prediction
    compose([big, pan_gt, pan_pr], ["RAW", "A", "B"]).save(
        out / f"{row.pair_id}_o0.png")
    compose([big, pan_pr, pan_gt], ["RAW", "A", "B"]).save(
        out / f"{row.pair_id}_o1.png")


def render(args):
    df = pd.read_csv(Path(args.out) / "pairs.csv")
    for row in df.itertuples():
        render_pair(row, args)
    print(f"rendered {2 * len(df)} PNGs under {Path(args.out) / 'crops'}")


# -------------------------------------------------------------------- query
def _png_b64(path):
    import base64
    return base64.standard_b64encode(path.read_bytes()).decode()


_ANT_WARNED = {"done": False}


def ask(client, png, args):
    """One judgement.  Returns the parsed answer plus usage.  Falls back to
    free-form JSON if the endpoint rejects structured output."""
    content = [
        {"type": "image", "source": {"type": "base64",
                                     "media_type": "image/png",
                                     "data": _png_b64(png)}},
        {"type": "text", "text": USER_PROMPT},
    ]
    kw = dict(
        model=args.model, max_tokens=args.max_tokens,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        output_config={"effort": args.effort,
                       "format": {"type": "json_schema",
                                  "schema": ANSWER_SCHEMA}},
    )
    import anthropic
    try:
        resp = client.messages.create(**kw)
    except anthropic.BadRequestError as e:
        if "format" not in str(e) and "output_config" not in str(e):
            raise
        if not _ANT_WARNED["done"]:
            _ANT_WARNED["done"] = True
            print(f"  structured output rejected, falling back to free "
                  f"JSON: {str(e)[:300]}")
        kw["output_config"] = {"effort": args.effort}
        resp = client.messages.create(**kw)
    text = next((b.text for b in resp.content if b.type == "text"), "")
    ans = _parse_answer(text)
    u = resp.usage
    return dict(
        ans, stop_reason=resp.stop_reason, request_id=resp._request_id,
        input_tokens=u.input_tokens, output_tokens=u.output_tokens,
        cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def _parse_answer(text):
    """Coerce whatever came back into the schema keys.  Models that ignore
    the schema tend to answer {"answer": "A"} or bare text."""
    ans = None
    try:
        ans = json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                ans = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    if not isinstance(ans, dict):
        ans = {"reason": "unparsable: " + text[:200]}
    choice = ans.get("choice", ans.get("answer", ans.get("verdict", "")))
    choice = str(choice).strip().strip('"').upper()[:3]
    choice = {"A": "A", "B": "B", "TIE": "tie"}.get(choice, "tie")
    try:
        conf = float(ans.get("confidence", np.nan))
    except (TypeError, ValueError):
        conf = np.nan
    try:
        n = int(ans.get("n_cells", -1))
    except (TypeError, ValueError):
        n = -1
    return {"choice": choice, "confidence": conf, "n_cells": n,
            "reason": str(ans.get("reason", ans.get("explanation", "")))[:300],
            "raw": text[:300]}


# Request variants for OpenAI-compatible servers, most capable first.  The
# first one the server accepts is remembered for the rest of the run, so a
# local vLLM/Ollama that rejects json_schema or reasoning_effort costs one
# failed call, not one per query.
_OAI_VARIANT = {"idx": None}


def _oai_variants(args):
    fmt_schema = {"type": "json_schema",
                  "json_schema": {"name": "verdict", "strict": True,
                                  "schema": ANSWER_SCHEMA}}
    fmt_obj = {"type": "json_object"}
    fmts = {"schema": (fmt_schema, fmt_obj, None), "object": (fmt_obj, None),
            "none": (None,)}[args.json_mode]
    v = []
    for fmt in fmts:
        for reasoning in ((args.effort,) if args.reasoning else ()) + (None,):
            kw = {}
            if fmt:
                kw["response_format"] = fmt
            if reasoning:
                kw["reasoning_effort"] = reasoning
            v.append(kw)
    return v


def ask_openai(client, png, args):
    import openai
    content = [
        {"type": "text", "text": USER_PROMPT + " Keys: choice (A, B or "
         "tie), confidence (0-1), n_cells (integer), reason (short)."},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + _png_b64(png),
            "detail": "high"}},
    ]
    base = dict(
        model=args.model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": content}],
    )
    variants = _oai_variants(args)
    start = _OAI_VARIANT["idx"] or 0
    resp = None
    for i in range(start, len(variants)):
        kw = dict(base, **variants[i])
        for tok_key in ("max_completion_tokens", "max_tokens"):
            try:
                resp = client.chat.completions.create(
                    **kw, **{tok_key: args.max_tokens})
                break
            except openai.BadRequestError as e:
                if tok_key == "max_completion_tokens" and \
                        "max_completion_tokens" in str(e):
                    continue      # old server: retry with max_tokens
                last = e
                break
        if resp is not None:
            if _OAI_VARIANT["idx"] != i:
                _OAI_VARIANT["idx"] = i
                print(f"  openai backend: using request variant {i} "
                      f"{sorted(variants[i])}")
            break
    if resp is None:
        raise last
    msg = resp.choices[0].message
    ans = _parse_answer(msg.content or "")
    u = resp.usage
    return dict(
        ans, stop_reason=resp.choices[0].finish_reason,
        request_id=getattr(resp, "id", ""),
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
        cache_read=0, cache_write=0,
    )


def make_client(args):
    if args.backend == "openai":
        import openai
        key = os.environ.get(args.api_key_env) or "EMPTY"
        base = args.base_url or os.environ.get("OPENAI_BASE_URL")
        headers = {}
        for h in args.header or []:
            name, _, value = h.partition(":")
            headers[name.strip()] = value.strip().replace(
                "{session}", f"vlm-judge-{uuid.uuid4().hex[:12]}")
        print(f"openai backend: base_url={base or 'api.openai.com'} "
              f"key={'set' if key != 'EMPTY' else 'EMPTY'} "
              f"headers={sorted(headers)}")
        return openai.OpenAI(api_key=key, base_url=base, max_retries=5,
                             timeout=args.timeout,
                             default_headers=headers or None), ask_openai
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; the SDK will fall back to an "
              "`ant auth login` profile if one exists.", file=sys.stderr)
    return anthropic.Anthropic(max_retries=5), ask


def query(args):
    client, ask_fn = make_client(args)
    out = Path(args.out)
    df = pd.read_csv(out / "pairs.csv")
    log = out / "responses.jsonl"
    done = set()
    if log.exists():
        for line in log.open():
            r = json.loads(line)
            done.add((r["pair_id"], r["ordering"], r["model"], r["effort"]))
    jobs = []
    for row in df.itertuples():
        for o in (0, 1):
            key = (row.pair_id, o, args.model, args.effort)
            if key not in done:
                jobs.append((row.pair_id, o))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(done)} cached, {len(jobs)} to query with {args.model} "
          f"effort={args.effort} workers={args.workers}")
    if args.dry_run or not jobs:
        return

    tot_in = tot_out = 0
    t0 = time.time()
    with log.open("a") as fh, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(ask_fn, client,
                          out / "crops" / f"{pid}_o{o}.png", args): (pid, o)
                for pid, o in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            pid, o = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  {pid} o{o}: {type(e).__name__}: {e}")
                continue
            r.update(pair_id=pid, ordering=o, model=args.model,
                     effort=args.effort, ts=time.time())
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            tot_in += r["input_tokens"] + r["cache_read"] + r["cache_write"]
            tot_out += r["output_tokens"]
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s  "
                      f"in={tot_in} out={tot_out}")
    pin, pout = PRICES.get(args.model, (0, 0))
    cost = (f"approx cost ${tot_in * pin / 1e6 + tot_out * pout / 1e6:.2f} "
            "(output includes thinking)" if pin else
            "(no price table for this model)")
    print(f"tokens in={tot_in} out={tot_out}  {cost}")


# ------------------------------------------------------------------- report
def tercile(s, name):
    # rank first so heavily tied values (many 48-px crops) still split
    return pd.qcut(s.rank(method="first"), 3,
                   labels=[f"{name}:low", f"{name}:mid", f"{name}:high"])


def report(args):
    out = Path(args.out)
    pairs = pd.read_csv(out / "pairs.csv").set_index("pair_id")
    rs = [json.loads(l) for l in (out / "responses.jsonl").open()]
    rs = pd.DataFrame(rs)
    if args.model:
        rs = rs[rs.model == args.model]
    if args.effort:
        rs = rs[rs.effort == args.effort]
    # keep the latest answer per (pair, ordering)
    rs = rs.sort_values("ts").groupby(["pair_id", "ordering"]).last()

    def pick(r):
        if r.choice == "tie":
            return "tie"
        # ordering 0: A = GT; ordering 1: A = pred
        a_is_gt = r.name[1] == 0
        return "gt" if (r.choice == "A") == a_is_gt else "pred"

    rs["pick"] = rs.apply(pick, axis=1)
    rs["chose_A"] = rs.choice == "A"
    w = rs.unstack("ordering")
    w = w[w["pick"].notna().all(axis=1)]
    d = pd.DataFrame(index=w.index)
    d["pick0"], d["pick1"] = w["pick"][0], w["pick"][1]
    d["conf"] = w["confidence"].min(axis=1)
    d["consistent"] = d.pick0 == d.pick1
    d["decided"] = (d.pick0 != "tie") & (d.pick1 != "tie")
    d["correct_both"] = (d.pick0 == "gt") & (d.pick1 == "gt")
    d["wrong_both"] = (d.pick0 == "pred") & (d.pick1 == "pred")
    d["n_cells"] = w["n_cells"][0]
    d = d.join(pairs)
    d["contrast_t"] = tercile(d.contrast, "contrast")
    d["density_t"] = tercile(d.density, "density")
    d["size_t"] = tercile(d.crop_px, "crop_px")
    d.to_csv(out / "judgements.csv")

    def summarize(g):
        gated = g[g.consistent & g.decided & (g.conf >= args.conf)]
        cons = g[g.consistent & g.decided]
        return pd.Series({
            "n": len(g),
            "acc_any": (g.pick0.eq("gt").mean() + g.pick1.eq("gt").mean()) / 2,
            "tie_rate": ((g.pick0 == "tie").mean() + (g.pick1 == "tie").mean()) / 2,
            "consistent": (g.consistent & g.decided).mean(),
            "acc_consistent": cons.correct_both.mean() if len(cons) else np.nan,
            "n_gated": len(gated),
            "acc_gated": gated.correct_both.mean() if len(gated) else np.nan,
            "flipped_wrong": g.wrong_both.mean(),
        })

    blocks = [("ALL", d.groupby(lambda _: "all")),
              ("subset", d.groupby("subset")),
              ("event", d.groupby("event")),
              ("contrast", d.groupby("contrast_t", observed=True)),
              ("density", d.groupby("density_t", observed=True)),
              ("crop size", d.groupby("size_t", observed=True))]
    tables = []
    for name, g in blocks:
        t = g.apply(summarize)
        t.index.name = name
        tables.append(t)
    rep = pd.concat([t.reset_index().rename(columns={t.index.name: "stratum"})
                     .assign(by=t.index.name) for t in tables])
    rep.to_csv(out / "report.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"position bias: chose A in {rs.chose_A.mean():.0%} of answers "
          f"(50% = none); n pairs = {len(d)}; conf gate >= {args.conf}")
    print("acc_any: per-answer accuracy vs GT (ties count as wrong); "
          "consistent: both orderings pick the same candidate; "
          "acc_gated: accuracy on consistent+confident pairs; "
          "flipped_wrong: confidently picked the prediction over GT in both "
          "orderings\n")
    for t in tables:
        print(t.round(3).to_string(), "\n")
    per_image_table(d, args, out)
    write_sheet(d, w, out)


def per_image_table(d, args, out):
    """Per-image vote: does the judge side with the prediction or the GT?
    Joined with predict.py's per-image F1 where report.csv exists, so an
    image the model failed on can be checked against the judge's verdict."""
    g = d.groupby(["subset", "image"])
    t = pd.DataFrame({
        "n": g.size(),
        "pro_gt": g.apply(lambda x: ((x.pick0 == "gt").mean()
                                     + (x.pick1 == "gt").mean()) / 2),
        "pro_pred": g.apply(lambda x: ((x.pick0 == "pred").mean()
                                       + (x.pick1 == "pred").mean()) / 2),
        "pro_gt_both": g.correct_both.mean(),
    }).reset_index()
    f1 = []
    for s in t.subset.unique():
        rep = Path("predictions") / f"{args.run}_{s}_t{args.tid}" / "report.csv"
        if rep.exists():
            r = pd.read_csv(rep)
            r["image"] = r.image.str.replace("_label", "", regex=False)
            r["subset"] = s
            f1.append(r[["subset", "image", "f1", "n_gts", "n_preds"]])
    if f1:
        t = t.merge(pd.concat(f1), on=["subset", "image"], how="left")
    t = t.sort_values(["pro_gt", "pro_gt_both"], ascending=False)
    t.to_csv(out / "per_image.csv", index=False)
    if len(t) > 1:
        print("per image, sorted by how often the judge sided with GT "
              "(top 15):")
        print(t.head(15).round(3).to_string(index=False), "\n")
        if "f1" in t:
            r = t[["pro_gt", "f1"]].dropna()
            if len(r) > 2:
                print(f"Spearman(pro_gt, f1) = "
                      f"{r.pro_gt.corr(r.f1, method='spearman'):.2f} "
                      f"over {len(r)} images\n")


def write_sheet(d, w, out):
    rows = []
    for pid, r in d.sort_values(["subset", "event"]).iterrows():
        reasons = " / ".join(f"o{o}: {w['reason'][o][pid]}" for o in (0, 1))
        tag = ("OK" if r.correct_both else
               "WRONG" if r.wrong_both else "mixed")
        rows.append(
            f"<div class=c><img src='crops/{pid}_o0.png'><br>"
            f"<b>{pid}</b> {r.event} | pick o0={r.pick0} o1={r.pick1} "
            f"conf={r.conf:.2f} <b class={tag}>{tag}</b><br>"
            f"<small>{reasons}</small></div>")
    html = ("<html><head><style>body{font-family:sans-serif;background:#222;"
            "color:#ddd}.c{margin:12px;max-width:1400px}img{max-width:100%}"
            ".OK{color:#6f6}.WRONG{color:#f66}.mixed{color:#fc6}</style>"
            "</head><body><p>Ordering 0 shown: A = ground truth, "
            "B = prediction.</p>" + "\n".join(rows) + "</body></html>")
    (out / "sheet.html").write_text(html)
    print(f"contact sheet: {out / 'sheet.html'}")


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["mine", "render", "query", "report", "all"])
    p.add_argument("--run", default="375-pious-sound",
                   help="predictions/<run>_<subset>_t<tid> is the candidate")
    p.add_argument("--tid", type=int, default=0)
    p.add_argument("--data", default=os.path.expanduser("~/datasets/scs_val"))
    p.add_argument("--subsets", nargs="+", default=SUBSETS)
    p.add_argument("--out", default=None,
                   help="default analyses/vlm_judge/<run>")
    p.add_argument("--seed", type=int, default=0)
    # mine
    p.add_argument("--max_images", type=int, default=60,
                   help="images scanned per subset before sampling events")
    p.add_argument("--n_per_subset", type=int, default=40)
    p.add_argument("--events", nargs="+", default=EVENTS, choices=EVENTS,
                   help="event types to mine; drop 'spurious' on sparsely "
                        "annotated sets where unlabeled cells are real")
    p.add_argument("--per_image", type=int, default=0,
                   help="instead: scan every image and take up to K events "
                        "from each (per-image verdict rates)")
    # render
    p.add_argument("--panel_px", type=int, default=448)
    p.add_argument("--line_px", type=int, default=2)
    # query
    p.add_argument("--backend", default="anthropic",
                   choices=["anthropic", "openai"],
                   help="openai = any OpenAI-compatible chat endpoint")
    p.add_argument("--model", default=None,
                   help="default claude-opus-5-5; required for --backend "
                        "openai")
    p.add_argument("--effort", default=None,
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="query: anthropic output_config.effort, or openai "
                        "reasoning_effort with --reasoning (default "
                        "medium); report: only score answers made at "
                        "this effort (default: all)")
    p.add_argument("--reasoning", action="store_true",
                   help="openai backend: send reasoning_effort=<effort> "
                        "(o-series / gpt-5 style models); dropped "
                        "automatically if the server rejects it")
    p.add_argument("--base_url", default=None,
                   help="openai backend; default $OPENAI_BASE_URL")
    p.add_argument("--api_key_env", default="OPENAI_API_KEY",
                   help="env var holding the key for --backend openai; "
                        "'EMPTY' is sent when unset (local servers)")
    p.add_argument("--header", action="append", metavar="NAME: VALUE",
                   help="openai backend: extra request header, repeatable; "
                        "'{session}' in the value becomes a random per-run "
                        "id (opencode.ai wants 'x-opencode-session: "
                        "{session}')")
    p.add_argument("--json_mode", default="schema",
                   choices=["schema", "object", "none"],
                   help="openai backend: strongest response_format to try "
                        "(some models echo a json_schema instead of "
                        "answering; 'object' or 'none' skips it)")
    p.add_argument("--max_tokens", type=int, default=2048,
                   help="raise for reasoning models that spend tokens "
                        "before the visible answer")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="openai backend per-request timeout, seconds")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="only send this many queries (smoke test)")
    p.add_argument("--dry_run", action="store_true")
    # report
    p.add_argument("--conf", type=float, default=0.7,
                   help="confidence gate for acc_gated")
    args = p.parse_args()
    args.out = args.out or f"analyses/vlm_judge/{args.run}"
    if args.model is None:
        if args.backend == "openai":
            p.error("--backend openai needs --model <name served by the "
                    "endpoint>")
        args.model = "claude-opus-5-5"
    if args.effort is None and args.stage != "report":
        args.effort = "medium"

    stages = (["mine", "render", "query", "report"] if args.stage == "all"
              else [args.stage])
    for s in stages:
        globals()[s](args)


if __name__ == "__main__":
    main()
