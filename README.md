# UCell
A small cell segmentation model with not so small generalizability.

Ref: https://doi.org/10.48550/arXiv.2604.00243

UCell is a ~15M-parameter recursive transformer. The same two-block
transformer is applied repeatedly to refine an internal representation, so
depth comes from recursion rather than from parameters — which is why the
model stays small while generalizing across imaging modalities.

## Requirements

**Python 3.11 or newer.**

**GPU is optional.** UCell is small enough to run comfortably on CPU; a GPU
mainly buys throughput. Measured on a single 383x512 two-channel image:

| checkpoint | parameters | peak GPU memory | GPU (L40S) | CPU |
| --- | --- | --- | --- | --- |
| `ucell-768.pt`  | 14.7M | 376 MiB | 0.2 s | 12 s |
| `ucell-1024.pt` | 26.4M | 525 MiB | 0.2 s | — |

## Installation

```bash
git clone https://github.com/jiyuuchc/ucell.git
cd ucell

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
```

`pip install -e .` pulls PyTorch from PyPI, which ships a default CUDA build.
If you need a specific CUDA version, or a CPU-only build, install torch first
from the [official index](https://pytorch.org/get-started/locally/) and then
install UCell:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

The package itself needs only the core dependencies. A few more are used by
the command-line scripts and by the example below:

```bash
pip install huggingface_hub    # weight download
pip install -e ".[train]"      # fine-tuning: lightning, wandb, datasets
```

Verify the install:

```bash
python -c "import ucell, torch; print(torch.__version__, torch.cuda.is_available())"
```

## Model weights

Pretrained checkpoints live on the Hugging Face Hub at
[jiyuuchc/ucell](https://huggingface.co/jiyuuchc/ucell):

| file | size | hidden size |
| --- | --- | --- |
| `ucell-768.pt`  | 56 MB  | 768  |
| `ucell-1024.pt` | 101 MB | 1024 |

```bash
huggingface-cli download jiyuuchc/ucell ucell-768.pt
```

## Minimal example

Segment one image and write an instance label mask:

```python
import numpy as np
import tifffile
import torch
from huggingface_hub import hf_hub_download

from ucell.dynamics import compute_masks
from ucell.frm import FRMWrapper
from ucell.utils import pad_channel, patcherize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Weights.  The checkpoint carries its own config, already set to the
#    21-recursion inference schedule.  Pass `overrides` to change it, e.g.
#    overrides={"model.L_cycles": 7} for a shallower, faster pass.
weights = hf_hub_download("jiyuuchc/ucell", "ucell-768.pt")
model = FRMWrapper.from_checkpoint(weights).eval().to(DEVICE)

# 2. Image.  Any 2D or channel-first multichannel array, scaled to [0, 1]
#    and padded to the three channels the patch embedding expects.
img = tifffile.imread("sample.tif").astype("float32")
img = pad_channel(img / (img.max() + 1e-5))

# 3. Run.  patcherize tiles into image_size patches and stitches the result
#    back, so any image size works.  The trailing 0 is the task id.
with torch.device(DEVICE):
    out = patcherize(model.inner.predict, GS=model.config.image_size)(img, 0)

flow, cell_prob = np.moveaxis(out[..., :2], -1, 0), out[..., 2]

# 4. Flow + cell probability -> instance labels.
mask = compute_masks(
    flow * 4.0,
    cell_prob,
    cellprob_threshold=-0.5,
    min_size=5,
    device=torch.device(DEVICE),
)

tifffile.imwrite("sample_mask.tif", mask.astype("uint16"))
print(f"{mask.max()} instances -> sample_mask.tif")
```

## Fine-tuning

We recommend this setting:

```bash
python train.py \
  --init ${BASE_MODEL} \
  --config config.py:train \
  --config.data_dir=${DATADIR} \
  --config.n_iters=1 \
  --config.epochs_per_iter=1024 \
  --config.ema_decay=0.95 \
  --config.lora.rank=16
```

Save your training data (`*.tif`, `*_label.tif`) under `${DATADIR}/train`.
`${BASE_MODEL}` is one of the checkpoints above.

---
