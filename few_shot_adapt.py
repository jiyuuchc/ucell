#!/usr/bin/env python3
"""Few-shot adaptation pipeline for ucell.

Steps
-----
1. Rebuild data_dir/train/ by sampling --n_shots images from data_dir/validation/
   and writing a metadata.csv compatible with data.py's ImageFolder path.
2. Run train.py with the supplied extra arguments.
3. Locate the final_ema.pt produced by that run and run predict.py against
   data_dir/validation/.
4. Run eval.py against data_dir/validation/.
5. Copy the resulting report.csv to --results_dir with a descriptive name.

Usage
-----
python few_shot_adapt.py \\
    --data_dir ~/datasets/musc \\
    --n_shots 5 \\
    --seed 0 \\
    -- \\
    --config config.py:train \\
    --config.model.hidden_size=768 \\
    --init checkpoints/final_ema.pt \\
    --config.epochs_per_iter=20480 \\
    --config.n_iters=1 \\
    --config.task_id=2 \\
    --config.lora.rank=16 \\
    --config.seed=0 \\
    --config.ema_decay=0.95

Everything after the bare -- is forwarded verbatim to the sub-processes.
Pipeline-level flags must come before --.
"""

import csv
import os
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "Root dataset directory.  Must contain validation/ with "
    "<stem>.tif / <stem>_label.tif pairs.  train/ will be rebuilt each run.",
)
flags.DEFINE_string(
    "results_dir", "predictions/results",
    "Directory for saved report CSVs.",
)
flags.DEFINE_integer(
    "n_shots", 5,
    "Number of images sampled from validation/ for training.",
)
flags.DEFINE_integer(
    "seed", 0,
    "Random seed for selecting training examples.",
)
flags.DEFINE_integer(
    "num_runs", 1,
    "Number of times to rerun the full pipeline.",
)
flags.DEFINE_integer(
    "task_id", 0,
    "task_id written into metadata.csv.",
)
flags.DEFINE_string(
    "predict_outputdir", "predictions/tmp",
    "--outputdir forwarded to predict.py.",
)
flags.DEFINE_string(
    "checkpoints_dir", "checkpoints",
    "Root directory where train.py writes checkpoints.",
)
flags.DEFINE_string(
    "output_prefix", None,
    "Prefix for the saved report filename.  "
    "Defaults to the basename of --data_dir (e.g. 'musc').",
)
flags.DEFINE_float(
    "sz", 35.0,
    "Fixed typical cell size (pixels) written into metadata.csv.",
)
flags.DEFINE_integer(
    "min_cells", 10,
    "Skip examples with fewer than this many cells in the label mask.",
)
flags.DEFINE_bool(
    "cleanup", True,
    "Whether to delete per-run prediction temp directories after report copy.",
)
flags.DEFINE_bool(
    "skip_rebuild", False,
    "Whether to skip train data rebuild.",
)


# ---------------------------------------------------------------------------
# Step 1 – rebuild train split
# ---------------------------------------------------------------------------

def rebuild_train_split(
    data_dir: Path,
    n_shots: int,
    seed: int,
    task_id: int,
    sz: float,
    min_cells: int,
) -> int:
    """Sample n_shots examples from validation/ and write them to train/."""
    train_dir = data_dir / "train"

    if not FLAGS.skip_rebuild:
        val_dir = data_dir / "validation"
        if not val_dir.exists():
            sys.exit(f"[error] Validation directory not found: {val_dir}")
    
        label_files = sorted(val_dir.glob("*_label.tif"))
        if not label_files:
            sys.exit(
                f"[error] No *_label.tif files found in {val_dir}.  "
                "Validation folder must contain paired <stem>.tif / <stem>_label.tif files."
            )
    
        eligible_label_files = label_files
        if min_cells > 0:
            import numpy as np
            import tifffile
    
            eligible_label_files = []
            for label_fn in label_files:
                mask = tifffile.imread(label_fn)
                n_cells = len(np.unique(mask)) - (1 if 0 in mask else 0)
                if n_cells >= min_cells:
                    eligible_label_files.append(label_fn)
    
            if not eligible_label_files:
                sys.exit(
                    f"[error] No validation examples satisfy --min_cells={min_cells}."
                )
    
        rng = random.Random(seed)
        chosen = rng.sample(
            eligible_label_files,
            min(n_shots, len(eligible_label_files)),
        )
        if len(chosen) < n_shots:
            print(
                f"[warn] Only {len(chosen)} examples available "
                f"(requested {n_shots}); using all of them."
            )
    
        # Rebuild train directory
        if train_dir.exists():
            shutil.rmtree(train_dir)
        train_dir.mkdir(parents=True)
    
        rows = []
        for label_fn in chosen:
            stem = label_fn.name.replace("_label.tif", "")
            img_fn = val_dir / f"{stem}.tif"
            if not img_fn.exists():
                print(f"[warn] Image not found for {label_fn.name!r}, skipping.")
                continue
    
            shutil.copy2(img_fn, train_dir / img_fn.name)
            shutil.copy2(label_fn, train_dir / label_fn.name)
    
            rows.append(
                {
                    "image_file_name": img_fn.name,
                    # absolute path so compute_flow works from any cwd
                    "masks_file_name": str((train_dir / label_fn.name).resolve()),
                    "task_id": task_id,
                    "sz": round(sz, 2),
                }
            )
    elif (train_dir/"metadata.csv").exists():
        return 0
    else:   
        rows = [{
            "image_file_name": fn.name.replace("_label",""),
            "masks_file_name": str(fn.resolve()),
            "task_id": task_id,
            "sz": sz,
            } for fn in train_dir.glob("*_label*")
        ]

    if not rows:
        sys.exit("[error] No valid image/label pairs were copied to train/.")

    meta_path = train_dir / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_file_name", "masks_file_name", "task_id", "sz"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[step 1] {len(rows)} training sample(s) → {train_dir}")
    print(f"         metadata.csv: {meta_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# Step 2 – train
# ---------------------------------------------------------------------------

def run_train(
    extra_argv: list,
    project_root: Path,
    data_dir: Path,
    checkpoints_dir: Path,
    task_id: int,
    run_seed: int,
    session_tag: str,
    run_no: int,
) -> Path:
    """Run train.py and return the path to the newly created final_ema.pt."""

    cp_root = checkpoints_dir
    cp_root.mkdir(parents=True, exist_ok=True)

    # Inject --config.data_dir / --config.task_id / --config.seed if missing.
    # Also inject an isolated --dir to avoid collisions across concurrent jobs.
    has_data_dir = any(
        re.search(r"--config\.data_dir", a) for a in extra_argv
    )
    has_task_id = any(
        re.search(r"--config\.task_id", a) for a in extra_argv
    )
    has_seed = any(
        re.search(r"--config\.seed", a) for a in extra_argv
    )
    has_dir = any(
        re.search(r"^--dir(?:=|$)", a) for a in extra_argv
    )
    train_argv = list(extra_argv)
    if not has_data_dir:
        train_argv.append(f"--config.data_dir={data_dir}")
    if not has_task_id:
        train_argv.append(f"--config.task_id={task_id}")
    if not has_seed:
        train_argv.append(f"--config.seed={run_seed}")
    if not has_dir:
        isolated_dir = cp_root / session_tag / f"run_{run_no:03d}"
        isolated_dir.mkdir(parents=True, exist_ok=True)
        train_argv.append(f"--dir={isolated_dir}")
    else:
        isolated_dir = cp_root

    cmd = [sys.executable, "-u", str(project_root / "train.py")] + train_argv
    print(f"\n[step 2] Training …\n  {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        sys.exit(f"[error] train.py exited with code {result.returncode}")

    # Detect checkpoint only inside the isolated train directory.
    candidates = sorted(
        isolated_dir.rglob("final_ema.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        sys.exit(
            "[error] No final_ema.pt found after training under "
            f"{isolated_dir}."
        )
    checkpoint = candidates[-1]

    print(f"[step 2] Checkpoint: {checkpoint}")
    return checkpoint


# ---------------------------------------------------------------------------
# Step 3 – predict
# ---------------------------------------------------------------------------

def run_predict(
    project_root: Path,
    checkpoint: Path,
    val_dir: Path,
    predict_outputdir: str,
    extra_argv: list,
):
    """Run predict.py against val_dir using the located checkpoint."""

    # Forward model-shape flags only
    forward_flags = [
        a for a in extra_argv
        if a.startswith("--config.model.") or a.startswith("--config.lora.")
    ]

    # Re-inject task_id as --task_id (predict.py uses a different flag name)
    task_id_flag = [f"--task_id={FLAGS.task_id}"]
    for i, a in enumerate(extra_argv):
        if a == "--config.task_id" and i + 1 < len(extra_argv):
            task_id_flag = [f"--task_id={extra_argv[i + 1]}"]
            break
        m = re.match(r"--config\.task_id=(\S+)", a)
        if m:
            task_id_flag = [f"--task_id={m.group(1)}"]
            break

    cmd = (
        [sys.executable, "-u", str(project_root / "predict.py")]
        + ["--model", str(checkpoint)]
        + forward_flags
        + task_id_flag
        + ["--datadir", str(val_dir)]
        + ["--outputdir", predict_outputdir]
    )
    print(f"\n[step 3] Predicting …\n  {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        sys.exit(f"[error] predict.py exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Step 4 – eval
# ---------------------------------------------------------------------------

def run_eval(project_root: Path, val_dir: Path, predict_outputdir: str):
    cmd = [
        sys.executable, "-u", str(project_root / "eval.py"),
        "--datadir", str(val_dir),
        "--outputdir", predict_outputdir,
    ]
    print(f"\n[step 4] Evaluating …\n  {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        sys.exit(f"[error] eval.py exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Step 5 – archive report
# ---------------------------------------------------------------------------

def collect_report(
    predict_outputdir: str,
    results_dir: str,
    report_name: str,
) -> Path:
    src = Path(predict_outputdir) / "report.csv"
    if not src.exists():
        sys.exit(f"[error] report.csv not found at {src}")

    dst_dir = Path(results_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / report_name
    shutil.copy2(src, dst)
    print(f"[step 5] Report saved → {dst}")
    return dst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_flag(extra_argv: list, pattern: str, default: str = "NA") -> str:
    """Return the first captured group from any matching arg in extra_argv."""
    for a in extra_argv:
        m = re.search(pattern, a)
        if m:
            return m.group(1)
    return default


def _find_project_root() -> Path:
    """Find project root by walking up from this script location."""
    here = Path(__file__).resolve().parent
    required = ("train.py", "predict.py", "eval.py")

    for candidate in [here, *here.parents]:
        if all((candidate / name).exists() for name in required):
            return candidate

    raise RuntimeError(
        "Could not find project root containing train.py, predict.py, eval.py"
    )


def _resolve_from_root(project_root: Path, value: str) -> Path:
    """Resolve path relative to project_root unless already absolute."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv):
    # argv[1:] contains any arguments that absl did not consume (extra_argv)
    extra_argv = list(argv[1:])

    if not FLAGS.data_dir:
        raise app.UsageError("--data_dir is required.")

    project_root = _find_project_root()

    data_dir = Path(FLAGS.data_dir).expanduser().resolve()
    val_dir = data_dir / "validation"
    checkpoints_dir = _resolve_from_root(project_root, FLAGS.checkpoints_dir)
    base_predict_outputdir = _resolve_from_root(
        project_root, FLAGS.predict_outputdir
    )
    results_dir = _resolve_from_root(project_root, FLAGS.results_dir)

    prefix = FLAGS.output_prefix or data_dir.name
    hidden_size = _extract_flag(extra_argv, r"hidden_size[=\s]+(\d+)")
    lora_rank   = _extract_flag(extra_argv, r"lora\.rank[=\s]+(\d+)", default="0")
    if FLAGS.num_runs < 1:
        raise app.UsageError("--num_runs must be >= 1.")

    all_reports = []
    session_tag = (
        f"job_{int(time.time())}_{os.getpid()}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    for run_idx in range(FLAGS.num_runs):
        run_no = run_idx + 1
        run_seed = FLAGS.seed + run_idx
        run_predict_outputdir = (
            base_predict_outputdir / session_tag / f"run_{run_no:03d}"
        )

        report_name = (
            f"{prefix}_{FLAGS.n_shots}shot"
            f"_h{hidden_size}_r{lora_rank}"
            f"_seed{run_seed}_{session_tag}_run{run_no:03d}.csv"
        )

        print(
            f"\n[run {run_no}/{FLAGS.num_runs}] "
            f"seed={run_seed}, outputdir={run_predict_outputdir}"
        )

        # Step 1
        rebuild_train_split(
            data_dir,
            FLAGS.n_shots,
            run_seed,
            FLAGS.task_id,
            FLAGS.sz,
            FLAGS.min_cells,
        )

        # Step 2
        checkpoint = run_train(
            extra_argv,
            project_root,
            data_dir,
            checkpoints_dir,
            FLAGS.task_id,
            run_seed,
            session_tag,
            run_no,
        )

        # Step 3
        run_predict(
            project_root,
            checkpoint,
            val_dir,
            str(run_predict_outputdir),
            extra_argv,
        )

        # Step 4
        run_eval(project_root, val_dir, str(run_predict_outputdir))

        # Step 5
        dst = collect_report(
            str(run_predict_outputdir),
            str(results_dir),
            report_name,
        )
        all_reports.append(dst)

        if FLAGS.cleanup and run_predict_outputdir.exists():
            shutil.rmtree(run_predict_outputdir)
            print(f"[cleanup] Removed tmp dir: {run_predict_outputdir}")

    print("\n[done] Reports:")
    for report in all_reports:
        print(f"  - {report}")


if __name__ == "__main__":
    app.run(main)
