from absl import app, flags
from typing import Any
from pathlib import Path
import numpy as np
import tifffile
import torch
import torch.nn as nn
import pandas as pd
from ml_collections import config_flags
from tqdm import tqdm
from ucell.frm import FRMWrapper, FRM
from ucell.utils import patcherize
from ucell.dynamics import compute_masks 
from ucell.metrics import LabelMetrics
from sklearn.metrics import jaccard_score, roc_auc_score

flags.DEFINE_string("model", None, "checkpointing location")
flags.DEFINE_string("datapath", None, "test data dir")
flags.DEFINE_string("logdir", "predictions", "logging dir")
flags.DEFINE_bool("log_predictions", False, "Whether to log predictions")
_CONFIG = config_flags.DEFINE_config_file("config", "config.py")

def load_model():
    config = _CONFIG.value
    config.model.forward_dtype = "float32"
    model = FRMWrapper(config).eval()

    cp_path = flags.FLAGS.model
    cp = torch.load(cp_path, weights_only=True)
    if "ema_model" in cp:
        model_ = nn.Module()
        model_.module = model
        model_.load_state_dict(cp['ema_model'], strict=False)
    elif "model" in cp:
        model.load_state_dict(cp['model'])
    else:
        model.load_state_dict(cp)

    model = model.to('cuda')
    
    return torch.compile(model.inner)

def format_image(img):
    img = img - img.min()
    img = img / img.max()
    if img.ndim == 2:
        img = img[..., None]
    assert img.ndim == 3
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 2:
        img = np.pad(img, [[0,0], [0,0], [0,1]])
    
    assert img.shape[-1] == 3

    return img

def run(_):
    model = load_model()
    datapath = Path(flags.FLAGS.datapath)
    outpath = Path(flags.FLAGS.logdir)
    outpath.mkdir(exist_ok=True, parents=True)

    results = []

    for label_fn in tqdm(datapath.glob("**/*_label.tif")):
        name = label_fn.name
        relative = label_fn.parent.relative_to(datapath)
        img_fn = label_fn.with_name(name.replace("_label", ""))

        gt_label = tifffile.imread(label_fn)
        img = format_image(tifffile.imread(img_fn))

        assert gt_label.ndim == 2
        assert gt_label.shape == img.shape[:2]
        
        with torch.device('cuda'):
            out = patcherize(model.predict, B=8)(img)
        flow = np.moveaxis(out[:, :, :2], -1, 0)
        cell_prob = out[:, :, 2]
        mask = compute_masks(flow, cell_prob, flow_threshold=0)

        assert mask.shape == gt_label.shape

        if flags.FLAGS.log_predictions:
            outpath_ = outpath / relative
            outpath_.mkdir(exist_ok=True, parents=True)
            tifffile.imwrite(outpath_/name.replace("_label", "_flow"), flow)
            tifffile.imwrite(outpath_/name.replace("_label", "_logits"), cell_prob)
            tifffile.imwrite(outpath_/name.replace("_label", "_mask"), mask.astype('uint16'))
        
        label_metric = LabelMetrics()
        label_metric.update(mask, gt_label)
        
        result = dict(
            image=str(label_fn.relative_to(datapath)),
            cellprob_ji = jaccard_score(gt_label.flat != 0, cell_prob.flat > 0),
            cellprob_auc = roc_auc_score(gt_label.flat != 0, cell_prob.flat),
        )
        result.update(result = label_metric.compute())

        results.append(result)
    
    df = pd.DataFrame.from_records(results)
    df.to_csv(outpath/"report.csv", index=False)

if __name__ == "__main__":
    app.run(run)


