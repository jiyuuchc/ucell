"""Convert a checkpoint into the portable {config, state_dict} format.

Older exports are pickled FRMWrapper objects, which freeze the module layout
into the file and break whenever that layout is refactored.  This rewrites one
into the portable format, which only depends on parameter names.

Training checkpoints (the Fabric dicts with step/model/optimizer) carry no
config, so pass --config to supply one; --key picks which state dict to take
out of them (default `model`, use `ema_model` for the EMA weights).

The recursion schedule stored in a training checkpoint is the one it was
trained under.  Depth is free at inference, so --override lets you write the
inference schedule into the exported file:

  python scripts/export_checkpoint.py ucell-768.pt out.pt \\
      --override model.L_cycles=21
"""

import argparse
import ast
from pathlib import Path

import ml_collections
import torch

from ucell.frm import FRMWrapper


def parse_override(text):
    """'model.L_cycles=21' -> ('model.L_cycles', 21)"""
    key, _, raw = text.partition("=")
    if not raw:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw  # plain string, e.g. forward_dtype=bfloat16
    return key.strip(), value


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("src", help="checkpoint to convert")
    ap.add_argument("dst", help="where to write the portable checkpoint")
    ap.add_argument("--config", default=None,
                    help="config.py[:preset], required when src carries none")
    ap.add_argument("--key", default="model",
                    help="state dict key inside a training checkpoint")
    ap.add_argument("--override", action="append", type=parse_override,
                    default=[], metavar="KEY=VALUE",
                    help="dotted config path to override; repeatable")
    args = ap.parse_args()

    overrides = dict(args.override)

    if args.config:
        # a training checkpoint: build from the given config, then load
        import importlib.util

        from ml_collections import config_flags  # noqa: F401  (import check)

        path, _, preset = args.config.partition(":")
        spec = importlib.util.spec_from_file_location("_cfg", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.get_config(preset) if preset else module.get_config()
        if overrides:
            config.update_from_flattened_dict(overrides)
        model = FRMWrapper(config).eval()
        model.load_checkpoint(f"{args.src}:{args.key}")
    else:
        model = FRMWrapper.from_checkpoint(args.src, overrides=overrides).eval()

    model.export(args.dst)

    check = torch.load(args.dst, weights_only=False, map_location="cpu")
    cfg = ml_collections.ConfigDict(check["config"])
    print(f"wrote {args.dst} ({Path(args.dst).stat().st_size / 2**20:.1f} MB)")
    print(f"  hidden_size {cfg.model.hidden_size}  depth {cfg.model.depth}  "
          f"H/L {cfg.model.H_cycles}/{cfg.model.L_cycles}  "
          f"norm_task_emb {cfg.model.norm_task_emb}")
    print(f"  {len(check['state_dict'])} tensors")


if __name__ == "__main__":
    main()
