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
from ucell.utils import patcherize, remove_small_instances
from ucell.dynamics import compute_masks, remove_bad_flow_masks
from ucell.metrics import LabelMetrics
from sklearn.metrics import jaccard_score, roc_auc_score

flags.DEFINE_string("model", None, "checkpointing location")
flags.DEFINE_string("datapath", None, "test data dir")
flags.DEFINE_string("logdir", "predictions", "logging dir")
flags.DEFINE_bool("log_predictions", False, "Whether to log predictions")
flags.DEFINE_bool("reuse", False, "Whether to reuse existing output")
flags.DEFINE_integer("niter", 500, "Num of integration steps for mask computation")
flags.DEFINE_float("flow_scaling", 5.0, "Flow scaling")
flags.DEFINE_float("cellprob_threshold", 0, "Cell prob logit threshold")
flags.DEFINE_float("min_amp", 0, "Min flow amptitude")
flags.DEFINE_float("flow_err_threshold", 0, "Flow error threshold") 
flags.DEFINE_integer("task_id", 0, "Task id for multi-task models")
flags.DEFINE_integer("min_area", 50, "Min cell area")

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


def _compute_masks(flow, cell_prob):
    if flags.FLAGS.min_amp > 0:
        flow_amp = (flow ** 2).sum(axis=0)
        flow = np.where(flow_amp >= flags.FLAGS.min_amp, flow, 0)
    mask = compute_masks(
        flow * flags.FLAGS.flow_scaling,
        cell_prob,
        niter=flags.FLAGS.niter, 
        cellprob_threshold=flags.FLAGS.cellprob_threshold,
        flow_threshold=0,
        min_size=flags.FLAGS.min_area,
        max_size_fraction=0.4,
    )
    
    if flags.FLAGS.flow_err_threshold > 0:
        mask = remove_bad_flow_masks(mask, flow * 5, threshold=flags.FLAGS.flow_err_threshold, device=torch.device('cuda'))

    return mask


def run(_):
    config = _CONFIG.value

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
        
        if flags.FLAGS.reuse:
            outpath_ = outpath / relative
            flow = tifffile.imread(outpath_/name.replace("_label", "_flow"))
            cell_prob = tifffile.imread(outpath_/name.replace("_label", "_logits"))
        else:
            with torch.device('cuda'):
                out = patcherize(
                    model.predict, 
                    GS=config.image_size, 
                    B=4 if config.image_size>=512 else 8
                )(img, flags.FLAGS.task_id)

            flow = np.moveaxis(out[:, :, :2], -1, 0)
            cell_prob = out[:, :, 2]

        mask = _compute_masks(flow, cell_prob)

        assert mask.shape == gt_label.shape

        if flags.FLAGS.log_predictions:
            outpath_ = outpath / relative
            outpath_.mkdir(exist_ok=True, parents=True)
            if not flags.FLAGS.reuse:
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
        result.update(label_metric.compute())

        results.append(result)
    
    df = pd.DataFrame.from_records(results)
    df.to_csv(outpath/"report.csv", index=False)

    print(df.drop("image", axis=1).mean().to_string())

if __name__ == "__main__":
    app.run(run)


