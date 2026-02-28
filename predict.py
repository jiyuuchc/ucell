from absl import app, flags
from pathlib import Path
import numpy as np
import tifffile
import torch
import torch.nn as nn
import pandas as pd
from ml_collections import config_flags
from tqdm import tqdm
from ucell.frm import FRMWrapper
from ucell.utils import patcherize
from ucell.dynamics import compute_masks, remove_bad_flow_masks

flags.DEFINE_string("model", None, "checkpointing location")
flags.DEFINE_string("datadir", None, "test data dir")
flags.DEFINE_string("outputdir", "predictions", "logging dir")
flags.DEFINE_integer("niter", 500, "Num of integration steps for mask computation")
flags.DEFINE_float("flow_scaling", 4.0, "Flow scaling")
flags.DEFINE_float("cellprob_threshold", 0, "Cell prob logit threshold")
flags.DEFINE_float("flow_err_threshold", 0, "Flow error threshold") 
flags.DEFINE_integer("task_id", 0, "Task id for multi-task models")
flags.DEFINE_integer("min_area", 5, "Min cell area")

_CONFIG = config_flags.DEFINE_config_file("config", "config.py")

def load_model():
    if flags.FLAGS.model.startswith("http"):
        return flags.FLAGS.model

    config = _CONFIG.value

    model = FRMWrapper(config).eval()

    default_lora_scaling = None
    if hasattr(config, "lora") and config.lora.rank > 0:
        default_lora_scaling = config.lora.alpha / config.lora.rank

    model.load_checkpoint(
        flags.FLAGS.model,
        default_lora_scaling=default_lora_scaling,
    )

    model = model.to('cuda')

    return torch.compile(model.inner)

def format_image(img):
    from ucell.utils import pad_channel
    # img = img - img.min()
    img = img / (img.max() + 1e-5)
    img = pad_channel(img)
    return img


def _compute_masks(flow, cell_prob):
    mask = compute_masks(
        flow * flags.FLAGS.flow_scaling,
        cell_prob,
        niter=flags.FLAGS.niter, 
        cellprob_threshold=flags.FLAGS.cellprob_threshold,
        flow_threshold=0,
        min_size=flags.FLAGS.min_area,
        max_size_fraction=0.4,
        device=torch.device('cuda')
    )

    if flags.FLAGS.flow_err_threshold > 0:
        mask = remove_bad_flow_masks(mask, flow * 5, threshold=flags.FLAGS.flow_err_threshold, device=torch.device('cuda'))

    return mask


def grpc_call(server, image):
    import grpc
    import biopb.image as proto
    from biopb.image.utils import serialize_from_numpy, deserialize_to_numpy

    request = proto.ProcessRequest(
        image_data = proto.ImageData(pixels=serialize_from_numpy(image)),
    )

    use_https = server.startswith("https://")
    server = server.strip("https://").lstrip("http://")
    options=[
        ('grpc.max_send_message_length', 1024**3),
        ('grpc.max_receive_message_length', 1024**3)
    ]
    if use_https:
        with grpc.secure_channel(target=server, credentials=grpc.ssl_channel_credentials(), options=options) as channel:        
            stub = proto.ProcessImageStub(channel)
            response = stub.Run(request)
    else:
        with grpc.insecure_channel(target=server, options=options) as channel:
            stub = proto.ProcessImageStub(channel)
            response = stub.Run(request)

    label = deserialize_to_numpy(response.image_data.pixels)
    
    return label.squeeze()

def run(_):
    config = _CONFIG.value

    model = load_model()
    datapath = Path(flags.FLAGS.datadir)
    outpath = Path(flags.FLAGS.outputdir)

    for label_fn in tqdm(datapath.glob("**/*_label.tif")):
        name = label_fn.name
        relative = label_fn.parent.relative_to(datapath)

        outpath_ = outpath / relative
        outpath_.mkdir(exist_ok=True, parents=True)
        # if (outpath_/name.replace("_label", "_mask")).exists():
        #     # print(f"Skipping {name} as output already exists")
        #     continue

        img_fn = label_fn.with_name(name.replace("_label", ""))
        img = format_image(tifffile.imread(img_fn))

        if isinstance(model, nn.Module):
            with torch.device('cuda'):
                out = patcherize(
                    model.predict, 
                    GS=config.image_size,
                )(img, flags.FLAGS.task_id)

            flow = np.moveaxis(out[:, :, :2], -1, 0)
            cell_prob = out[:, :, 2]

            mask = _compute_masks(flow, cell_prob)

        else:
            flow, cell_prob = None, None
            mask = grpc_call(model, img)

        if flow is not None:
            tifffile.imwrite(outpath_/name.replace("_label", "_flow"), flow)
        if cell_prob is not None:
            tifffile.imwrite(outpath_/name.replace("_label", "_logits"), cell_prob)
        tifffile.imwrite(outpath_/name.replace("_label", "_mask"), mask.astype('uint16'))


if __name__ == "__main__":
    app.run(run)


